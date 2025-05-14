from fastapi import APIRouter, UploadFile, HTTPException, BackgroundTasks, File
from ...utils.file_handlers import save_upload_file
from ...core.yolo_detector import LogoDetector
from ...core.llm_analyzer import LLMAnalyzer
from typing import Dict, Any
import os
from pathlib import Path
from app.utils.logger import setup_logger

router = APIRouter()
logo_detector = LogoDetector()
llm_analyzer = LLMAnalyzer()
logger = setup_logger("video_endpoint")

# 분석 결과를 저장할 임시 저장소
analysis_results: Dict[str, Any] = {}

@router.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    비디오 파일을 업로드하고 로고 분석을 시작합니다.
    
    Args:
        background_tasks (BackgroundTasks): 백그라운드 작업 관리자
        file (UploadFile): 업로드할 비디오 파일
        
    Returns:
        dict: 업로드 결과 정보
    """
    logger.info(f"비디오 업로드 시작: {file.filename}")
    
    try:
        # 임시 파일 저장
        temp_path = Path("temp") / file.filename
        temp_path.parent.mkdir(exist_ok=True)
        
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"비디오 파일 저장 완료: {temp_path}")
        
        # 초기 상태 설정
        analysis_results[file.filename] = {
            "status": "processing",
            "message": "분석이 시작되었습니다."
        }
        
        # 백그라운드에서 로고 감지 및 분석 수행
        background_tasks.add_task(process_video, str(temp_path), file.filename)
        
        return {
            "message": "파일이 성공적으로 업로드되었습니다. 분석이 시작되었습니다.",
            "filename": file.filename
        }
    except Exception as e:
        logger.error(f"비디오 업로드 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analysis/{filename}")
async def get_analysis(filename: str):
    """
    비디오 분석 결과를 조회합니다.
    
    Args:
        filename (str): 분석 결과를 조회할 파일 이름
        
    Returns:
        dict: 분석 결과
    """
    if filename not in analysis_results:
        raise HTTPException(status_code=404, detail="분석 결과를 찾을 수 없습니다.")
    
    result = analysis_results[filename]
    if result.get("status") == "processing":
        raise HTTPException(status_code=202, detail="분석이 진행 중입니다.")
    
    return {
        "analysis": result.get("analysis", {}),
        "status": result.get("status"),
        "message": result.get("message")
    }

@router.get("/status/{filename}")
async def get_processing_status(filename: str):
    """
    비디오 처리 상태를 조회합니다.
    
    Args:
        filename (str): 상태를 조회할 파일 이름
        
    Returns:
        dict: 처리 상태 정보
    """
    if filename not in analysis_results:
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    
    return {
        "status": analysis_results[filename].get("status", "unknown"),
        "filename": filename,
        "message": analysis_results[filename].get("message", "")
    }

async def process_video(file_path: str, filename: str):
    """
    비디오 파일을 처리하여 로고를 감지하고 분석합니다.
    
    Args:
        file_path (str): 처리할 비디오 파일 경로
        filename (str): 처리할 파일 이름
    """
    try:
        # YOLO 감지
        detector = LogoDetector()
        detections = detector.detect_logos(Path(file_path))
        
        # LLM 분석
        analyzer = LLMAnalyzer()
        analysis = analyzer.analyze_detections(detections)
        
        # 결과 저장
        analysis_results[filename] = {
            "status": "completed",
            "message": "분석이 완료되었습니다.",
            "detections": detections,
            "analysis": analysis
        }
        
        # 임시 파일 삭제
        Path(file_path).unlink()
        logger.info("임시 파일 삭제 완료")
        
        logger.info("비디오 처리 완료")
    except Exception as e:
        logger.error(f"비디오 처리 중 오류 발생: {str(e)}")
        analysis_results[filename] = {
            "status": "error",
            "message": f"처리 중 오류가 발생했습니다: {str(e)}"
        } 