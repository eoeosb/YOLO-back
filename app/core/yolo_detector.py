from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any
import os
from app.utils.logger import setup_logger

logger = setup_logger("yolo_detector")

class LogoDetector:
    def __init__(self, model_path: str = "best.pt"):
        """
        YOLOv11n 모델을 초기화합니다.

        Args:
            model_path (str): YOLO 모델 파일 경로
        """
        self.model = YOLO(model_path)
        self.conf_threshold = 0.25  # 신뢰도 임계값
        self.iou_threshold = 0.45   # IoU 임계값
        logger.info(f"YOLO 모델 로드 완료: {model_path}")

    def detect_logos(self, video_path: Path, sample_rate: int = 30, save_images: bool = True, output_dir: str = "output") -> List[Dict[str, Any]]:
        """
        비디오에서 로고를 감지합니다.

        Args:
            video_path (Path): 비디오 파일 경로
            sample_rate (int): 프레임 샘플링 간격 (기본값: 30프레임마다)
            save_images (bool): 감지된 로고가 있는 이미지를 저장할지 여부
            output_dir (str): 결과 이미지를 저장할 디렉토리

        Returns:
            List[Dict[str, Any]]: 감지된 로고 정보 리스트
        """
        logger.info(f"비디오 로고 감지 시작: {video_path}")
        logger.info(f"샘플링 레이트: {sample_rate} 프레임")
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        detections = []
        frame_count = 0
        
        # 결과 이미지 저장 디렉토리 생성
        if save_images:
            os.makedirs(output_dir, exist_ok=True)
            video_name = Path(video_path).stem
            frames_dir = os.path.join(output_dir, video_name)
            os.makedirs(frames_dir, exist_ok=True)
            logger.info(f"결과 이미지 저장 디렉토리: {frames_dir}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                logger.info(f"프레임 {frame_count} 처리 중...")
                results = self.model(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                
                frame_detections = []  # 현재 프레임에서 감지된 로고들

                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        confidence = float(box.conf[0])
                        if confidence > self.conf_threshold:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            class_id = int(box.cls[0])
                            class_name = result.names[class_id]

                            detection = {
                                "timestamp": frame_count / fps,
                                "class": class_name,
                                "confidence": confidence,
                                "bbox": [x1, y1, x2, y2],
                                "frame_number": frame_count,
                                "size": (x2 - x1) * (y2 - y1)
                            }
                            detections.append(detection)
                            frame_detections.append(detection)
                            logger.debug(f"감지된 로고: {detection}")
                
                # 로고가 감지된 프레임에 바운딩 박스를 그리고 이미지 저장
                if save_images and frame_detections:
                    annotated_frame = frame.copy()
                    for detection in frame_detections:
                        x1, y1, x2, y2 = detection["bbox"]
                        class_name = detection["class"]
                        confidence = detection["confidence"]
                        
                        # 바운딩 박스 그리기
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 클래스 이름과 신뢰도 텍스트 추가
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # 이미지 저장
                    img_path = os.path.join(frames_dir, f"frame_{frame_count:06d}.jpg")
                    cv2.imwrite(img_path, annotated_frame)
                    logger.info(f"감지 결과 이미지 저장: {img_path}")

            frame_count += 1

        cap.release()
        logger.info(f"비디오 로고 감지 완료. 총 {len(detections)}개의 로고 감지됨")
        return detections
