import os
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from app.utils.logger import setup_logger

load_dotenv()

logger = setup_logger("llm_analyzer")

class LLMAnalyzer:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        OpenAI API 클라이언트를 초기화합니다.
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        logger.info(f"LLM 분석기 초기화 완료: {model_name}")
        
    def analyze_detections(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        감지된 로고들을 분석하여 PPL 효과를 평가합니다.
        
        Args:
            detections (List[Dict[str, Any]]): YOLO로 감지된 로고 정보 리스트
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        logger.info("로고 감지 결과 분석 시작")
        logger.info(f"분석할 감지 결과 수: {len(detections)}")
        
        # 로고별로 감지 정보 그룹화
        logo_groups = {}
        for detection in detections:
            logo_name = detection["class"]
            if logo_name not in logo_groups:
                logo_groups[logo_name] = []
            logo_groups[logo_name].append(detection)
        
        # 각 로고별 분석 수행
        analysis_results = {}
        for logo_name, logo_detections in logo_groups.items():
            # 로고 노출 시간 계산
            total_duration = sum(d["timestamp"] for d in logo_detections)
            # 노출 빈도 계산
            frequency = len(logo_detections)
            # 평균 신뢰도 계산
            avg_confidence = sum(d["confidence"] for d in logo_detections) / len(logo_detections)
            
            # LLM에 분석 요청
            prompt = f"""
            다음은 드라마/예능 프로그램에서 감지된 {logo_name} 브랜드의 PPL 정보입니다:
            - 총 노출 시간: {total_duration:.2f}초
            - 노출 빈도: {frequency}회
            - 평균 신뢰도: {avg_confidence:.2f}
            
            이 정보를 바탕으로 다음을 분석해주세요:
            1. PPL 효과성 (1-10점)
            2. 주요 노출 시점과 특징
            3. 개선 제안사항
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "당신은 PPL 효과를 분석하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            analysis_results[logo_name] = {
                "total_duration": total_duration,
                "frequency": frequency,
                "avg_confidence": avg_confidence,
                "llm_analysis": response.choices[0].message.content
            }
        
        logger.info("로고 감지 결과 분석 완료")
        logger.debug(f"분석 결과: {analysis_results}")
        
        return analysis_results 