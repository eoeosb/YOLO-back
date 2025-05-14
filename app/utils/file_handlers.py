import os
from pathlib import Path
from fastapi import UploadFile
from ..config.settings import ALLOWED_VIDEO_EXTENSIONS, VIDEO_DIR, MAX_VIDEO_SIZE

async def save_upload_file(upload_file: UploadFile) -> Path:
    """
    업로드된 비디오 파일을 저장합니다.
    
    Args:
        upload_file (UploadFile): 업로드된 파일 객체
        
    Returns:
        Path: 저장된 파일의 경로
        
    Raises:
        ValueError: 파일 확장자가 허용되지 않거나 파일 크기가 너무 큰 경우
    """
    # 파일 확장자 검사
    file_extension = Path(upload_file.filename).suffix.lower()
    if file_extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError(f"허용되지 않는 파일 형식입니다. 허용된 형식: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}")
    
    # 파일 크기 검사
    file_size = 0
    chunk_size = 1024 * 1024  # 1MB
    
    # 임시 파일로 저장
    temp_file_path = VIDEO_DIR / f"temp_{upload_file.filename}"
    with open(temp_file_path, "wb") as f:
        while chunk := await upload_file.read(chunk_size):
            file_size += len(chunk)
            if file_size > MAX_VIDEO_SIZE:
                os.remove(temp_file_path)
                raise ValueError(f"파일 크기가 너무 큽니다. 최대 크기: {MAX_VIDEO_SIZE/1024/1024}MB")
            f.write(chunk)
    
    # 파일 이름 중복 방지를 위한 처리
    final_file_path = VIDEO_DIR / upload_file.filename
    if final_file_path.exists():
        base_name = final_file_path.stem
        counter = 1
        while final_file_path.exists():
            final_file_path = VIDEO_DIR / f"{base_name}_{counter}{file_extension}"
            counter += 1
    
    # 임시 파일을 최종 위치로 이동
    os.rename(temp_file_path, final_file_path)
    
    return final_file_path 