from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pathlib import Path
import os
import shutil
import uuid
import json
from datetime import datetime
from app.core.yolo_detector import LogoDetector
from app.core.llm_analyzer import LLMAnalyzer
from app.utils.logger import setup_logger

router = APIRouter()
logo_detector = LogoDetector()
llm_analyzer = LLMAnalyzer()

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
    
    # 백그라운드 작업으로 처리
    background_tasks.add_task(process_video, upload_path, sample_rate, file_id, file.filename)
    
    return JSONResponse(content={
        "message": "비디오 분석이 시작되었습니다.",
        "file_id": file_id
    })

# 기존 /analysis/{filename} 엔드포인트도 복원 (기존 클라이언트 호환성 유지)
@router.get("/analysis/{file_id}")
async def get_analysis(file_id: str):
    """
    기존 클라이언트와의 호환성을 위해 /analysis 엔드포인트 유지
    /result 엔드포인트와 동일한 기능
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
        
    # 처리중인지 확인
    result_path = RESULTS_DIR / f"{file_id}.json"
    if result_path.exists():
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data.get("status") == "processing":
                return JSONResponse(
                    status_code=202,
                    content={
                        "status": "processing",
                        "message": "분석이 진행 중입니다. 잠시 후 다시 시도해주세요."
                    }
                )
    
    return await get_analysis_result(file_id)

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

async def process_video(video_path: Path, sample_rate: int, file_id: str, original_filename: str):
    """
    비디오를 처리하고 결과를 저장하는 백그라운드 작업
    """
    logger.info(f"비디오 처리 시작: {video_path}, 샘플링 레이트: {sample_rate}")
    result_path = RESULTS_DIR / f"{file_id}.json"
    
    try:
        # 처리 상태 저장
        save_result_status(result_path, "processing", "비디오 분석 진행 중...")
        
        # 로고 감지
        detections = logo_detector.detect_logos(video_path, sample_rate)
        logger.info(f"로고 감지 완료: {len(detections)}개 감지됨")
        
        # LLM 분석 (비디오 경로 전달하여 오디오 추출 및 분석 활성화)
        analysis_results = llm_analyzer.analyze_detections(detections, video_path)
        logger.info(f"LLM 분석 완료: {len(analysis_results)}개 로고 분석됨")
        
        # 결과 저장 - 프론트엔드와 호환되는 형식
        result = {
            "status": "completed",
            "message": "비디오 분석이 완료되었습니다.",
            "file_id": file_id,
            "filename": file_id,  # 프론트엔드 호환성
            "original_filename": original_filename,
            "analysis_time": datetime.now().isoformat(),
            "sample_rate": sample_rate,
            "total_detections": len(detections),
            "analysis_results": analysis_results,
            "analysis": analysis_results  # 프론트엔드 호환성
        }
        
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        logger.info(f"분석 결과 저장 완료: {result_path}")
        
    except Exception as e:
        logger.error(f"비디오 처리 중 오류 발생: {e}")
        save_result_status(result_path, "error", f"처리 중 오류가 발생했습니다: {str(e)}")
    finally:
        # 임시 비디오 파일 삭제 여부 선택
        # 필요에 따라 주석 해제
        # if video_path.exists():
        #     os.unlink(video_path)
        #     logger.info(f"임시 비디오 파일 삭제 완료: {video_path}")
        pass

def save_result_status(result_path: Path, status: str, message: str):
    """
    분석 상태 정보를 JSON 파일로 저장합니다.
    """
    result = {
        "status": status,
        "message": message,
        "update_time": datetime.now().isoformat()
    }
    
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"상태 정보 저장 완료: {result_path} ({status})") 