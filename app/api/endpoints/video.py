from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse
from pathlib import Path
import os
import shutil
import uuid
import json
import asyncio
from datetime import datetime
from app.core.yolo_detector import LogoDetector
from app.core.llm_analyzer import LLMAnalyzer
from app.core.websocket_manager import WebSocketManager
from app.utils.logger import setup_logger

router = APIRouter()
logo_detector = LogoDetector()
llm_analyzer = LLMAnalyzer()
websocket_manager = WebSocketManager()

logger = setup_logger("video_router")

UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 기존 /upload 엔드포인트 복원 (기존 클라이언트 호환성 유지)
@router.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sample_rate: int = Form(30)
):
    """
    기존 클라이언트와의 호환성을 위해 /upload 엔드포인트 유지
    /analyze 엔드포인트와 동일한 기능
    """
    response = await analyze_video(background_tasks, file, sample_rate)
    
    # 프론트엔드에서 사용하기 쉽게 filename 필드 추가
    content = response.body.decode()
    data = json.loads(content)
    data["filename"] = data.get("file_id")  # 기존 클라이언트 호환성을 위해 filename 필드 추가
    
    return JSONResponse(content=data)

@router.post("/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sample_rate: int = Form(30)
):
    # 업로드 파일명 생성 (고유 ID)
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    upload_path = UPLOAD_DIR / f"{file_id}{file_extension}"
    
    logger.info(f"비디오 업로드 시작: {file.filename} -> {upload_path}")
    
    # 파일 저장
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"파일 저장 실패: {e}")
        raise HTTPException(status_code=500, detail=f"파일 저장 실패: {str(e)}")
    finally:
        file.file.close()
    
    logger.info(f"비디오 업로드 완료: {upload_path}")
    
    # 초기 상태 파일 생성
    result_path = RESULTS_DIR / f"{file_id}.json"
    initial_status = {
        "status": "waiting",
        "message": "WebSocket 연결 대기 중...",
        "progress": 0,
        "update_time": datetime.now().isoformat()
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(initial_status, f, ensure_ascii=False, indent=2)
    
    # 클라이언트가 WebSocket을 연결할 수 있도록 잠시 대기 후 분석 시작
    async def delayed_process():
        await asyncio.sleep(1)  # WebSocket 연결을 위한 대기
        await process_video(upload_path, sample_rate, file_id, file.filename)
    
    background_tasks.add_task(delayed_process)
    
    return JSONResponse(content={
        "message": "비디오 분석이 곧 시작됩니다. WebSocket으로 진행 상황을 확인하세요.",
        "file_id": file_id
    })

# 기존 /analysis/{filename} 엔드포인트도 복원 (기존 클라이언트 호환성 유지)
@router.get("/analysis/{file_id}")
async def get_analysis(file_id: str):
    """
    분석 결과를 조회합니다.
    """
    # 'undefined'가 전달된 경우 적절한 오류 메시지 반환
    if file_id == "undefined":
        logger.warning("클라이언트가 'undefined'를 file_id로 전송했습니다.")
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "잘못된 file_id: 'undefined'가 전달되었습니다. 파일을 먼저 업로드하고 반환된 file_id를 사용하세요."
            }
        )
    
    # 결과 파일 경로
    result_path = RESULTS_DIR / f"{file_id}.json"
    logger.info(f"결과 파일 조회: {result_path}")
    
    if not result_path.exists():
        logger.error(f"결과 파일을 찾을 수 없음: {result_path}")
        raise HTTPException(status_code=404, detail="분석 결과를 찾을 수 없습니다.")
    
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)
            logger.debug(f"로드된 결과: {result}")
            
        # 결과 구조 확인 및 수정
        if "analysis_results" in result:
            result["analysis"] = result["analysis_results"]
        
        # 상태 정보가 없는 경우 추가
        if "status" not in result:
            if "analysis" in result and result["analysis"]:
                result["status"] = "completed"
            else:
                result["status"] = "processing"
        
        # WebSocket 연결 필요 여부 추가
        result["needs_websocket"] = result.get("status") not in ["completed", "error"]
        
        logger.info(f"결과 반환: status={result.get('status')}, has_analysis={bool(result.get('analysis'))}")
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"결과 파일 읽기 실패: {e}")
        raise HTTPException(status_code=500, detail="결과 파일을 읽는 중 오류가 발생했습니다.")

@router.get("/result/{file_id}")
async def get_analysis_result(file_id: str):
    # 'undefined'가 전달된 경우 적절한 오류 메시지 반환
    if file_id == "undefined":
        logger.warning("클라이언트가 'undefined'를 file_id로 전송했습니다.")
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "잘못된 file_id: 'undefined'가 전달되었습니다. 파일을 먼저 업로드하고 반환된 file_id를 사용하세요."
            }
        )
    
    # 결과 파일 경로
    result_path = RESULTS_DIR / f"{file_id}.json"
    
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="분석 결과가 아직 준비되지 않았습니다.")
    
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        
        # 프론트엔드 호환성을 위해 응답 구조 조정
        if "analysis_results" in result:
            result["analysis"] = result.get("analysis_results")
            
        # 기존 클라이언트 호환성을 위해 필요한 경우 필드 추가
        if "filename" not in result and "file_id" in result:
            result["filename"] = result["file_id"]
        
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"결과 파일 읽기 실패: {e}")
        raise HTTPException(status_code=500, detail="결과 파일을 읽는 중 오류가 발생했습니다.")

@router.websocket("/ws/{file_id}")
async def websocket_endpoint(websocket: WebSocket, file_id: str):
    # 먼저 현재 상태 확인
    result_path = RESULTS_DIR / f"{file_id}.json"
    if result_path.exists():
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                current_status = json.load(f)
                if current_status.get("status") in ["completed", "error"]:
                    # 이미 처리가 완료된 경우 연결을 수락한 후 바로 완료 메시지 전송하고 정상적으로 종료
                    await websocket.accept()
                    await websocket.send_json({
                        "status": current_status["status"],
                        "message": current_status["message"],
                        "progress": 100 if current_status["status"] == "completed" else -1
                    })
                    await websocket.close(code=1000)
                    return
        except Exception as e:
            logger.error(f"상태 파일 읽기 실패: {e}")
            await websocket.accept()
            await websocket.close(code=1011)  # 서버 내부 오류
            return
    
    # 처리가 진행 중이거나 시작되지 않은 경우 연결 수락
    await websocket_manager.connect(websocket, file_id)
    try:
        while True:
            await websocket.receive_text()
    except Exception as e:
        logger.error(f"WebSocket 오류 발생: {e}")
    finally:
        websocket_manager.disconnect(websocket, file_id)

async def process_video(video_path: Path, sample_rate: int, file_id: str, original_filename: str):
    """
    비디오를 처리하고 결과를 저장하는 백그라운드 작업
    """
    logger.info(f"비디오 처리 시작: {video_path}, 샘플링 레이트: {sample_rate}")
    result_path = RESULTS_DIR / f"{file_id}.json"
    
    try:
        # WebSocket 연결 대기
        logger.info(f"WebSocket 연결 대기 중: {file_id}")
        connection_success = await websocket_manager.wait_for_connection(file_id, timeout=10.0)
        
        if not connection_success:
            logger.warning(f"WebSocket 연결 대기 시간 초과: {file_id}")
        
        # 처리 상태 저장 및 진행상황 전송
        await update_progress(file_id, "processing", "비디오 분석 시작...", 0)
        
        # 로고 감지
        await update_progress(file_id, "processing", "로고 감지 중...", 20)
        detections = logo_detector.detect_logos(video_path, sample_rate)
        logger.info(f"로고 감지 완료: {len(detections)}개 감지됨")
        
        # LLM 분석 준비
        await update_progress(file_id, "processing", "음성 추출 및 분석 준비 중...", 40)
        
        # LLM 분석
        await update_progress(file_id, "processing", "로고 및 음성 컨텍스트 분석 중...", 60)
        analysis_results = llm_analyzer.analyze_detections(detections, video_path)
        logger.info(f"LLM 분석 완료: {len(analysis_results)}개 로고 분석됨")
        
        await update_progress(file_id, "processing", "결과 저장 중...", 80)
        
        # 결과 저장
        result = {
            "status": "completed",
            "message": "비디오 분석이 완료되었습니다.",
            "file_id": file_id,
            "filename": file_id,
            "original_filename": original_filename,
            "analysis_time": datetime.now().isoformat(),
            "sample_rate": sample_rate,
            "total_detections": len(detections),
            "analysis_results": analysis_results,
            "analysis": analysis_results  # 프론트엔드 호환성을 위해 두 필드 모두 유지
        }
        
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        logger.info(f"분석 결과 저장 완료: {result_path}")
        await update_progress(file_id, "completed", "분석이 완료되었습니다.", 100)
        
    except Exception as e:
        logger.error(f"비디오 처리 중 오류 발생: {e}")
        await update_progress(file_id, "error", f"처리 중 오류가 발생했습니다: {str(e)}", -1)
    finally:
        pass

async def update_progress(file_id: str, status: str, message: str, progress: int):
    """
    진행 상황을 저장하고 WebSocket으로 전송합니다.
    """
    result_path = RESULTS_DIR / f"{file_id}.json"
    
    try:
        # 기존 결과 파일이 있다면 읽어옴
        if result_path.exists():
            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)
        else:
            result = {}
            
        # 상태 정보 업데이트
        result.update({
            "status": status,
            "message": message,
            "progress": progress,
            "update_time": datetime.now().isoformat()
        })
        
        # 결과 파일에 저장
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # WebSocket으로 진행상황 전송
        await websocket_manager.broadcast_progress(file_id, {
            "status": status,
            "message": message,
            "progress": progress,
            "update_time": datetime.now().isoformat()
        })
        
        logger.info(f"진행상황 업데이트: {message} ({progress}%)")
    except Exception as e:
        logger.error(f"진행상황 업데이트 실패: {e}")
        # 에러가 발생해도 WebSocket 메시지는 전송 시도
        try:
            await websocket_manager.broadcast_progress(file_id, {
                "status": "error",
                "message": f"진행상황 업데이트 중 오류 발생: {str(e)}",
                "progress": -1,
                "update_time": datetime.now().isoformat()
            })
        except:
            pass 