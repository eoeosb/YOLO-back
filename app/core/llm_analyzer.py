import os
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from app.utils.logger import setup_logger
import tempfile
import subprocess
import shutil
from pathlib import Path

load_dotenv()

logger = setup_logger("llm_analyzer")

class LLMAnalyzer:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        OpenAI API 클라이언트를 초기화합니다.
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        logger.info(f"LLM 분석기 초기화 완료: {model_name}")
        
    def extract_audio(self, video_path: Path) -> str:
        """
        비디오에서 오디오를 추출하여 임시 파일로 저장합니다.
        
        Args:
            video_path (Path): 비디오 파일 경로
            
        Returns:
            str: 추출된 오디오 파일 경로
        """
        logger.info(f"비디오에서 오디오 추출 시작: {video_path}")
        
        # 비디오 파일 존재 확인
        if not os.path.exists(video_path):
            logger.error(f"비디오 파일을 찾을 수 없음: {video_path}")
            return None
        
        # 임시 파일 생성
        temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        # FFmpeg 경로 확인
        ffmpeg_path = "ffmpeg"  # 기본 PATH에서 찾기
        
        # Windows에서 일반적인 FFmpeg 설치 위치 확인
        possible_paths = [
            "C:\\ffmpeg\\bin\\ffmpeg.exe",
            "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs\\ffmpeg\\bin\\ffmpeg.exe")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                ffmpeg_path = path
                logger.info(f"FFmpeg 발견: {ffmpeg_path}")
                break
        
        # FFmpeg를 사용하여 오디오 추출
        try:
            # FFmpeg 존재 확인
            if shutil.which(ffmpeg_path) is None and not os.path.exists(ffmpeg_path):
                logger.error(f"FFmpeg를 찾을 수 없음. 오디오 추출 건너뜀.")
                return None
                
            cmd = [
                ffmpeg_path, "-y", "-i", str(video_path), 
                "-q:a", "0", "-map", "a", "-vn", temp_audio_path
            ]
            
            logger.info(f"FFmpeg 명령어 실행: {' '.join(cmd)}")
            
            process = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"오디오 추출 완료: {temp_audio_path}")
            
            if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                return temp_audio_path
            else:
                logger.error("오디오 파일이 생성되지 않았거나 크기가 0입니다.")
                return None
                
        except subprocess.CalledProcessError as e:
            logger.error(f"오디오 추출 실패 (FFmpeg 오류): {e}")
            logger.error(f"STDERR: {e.stderr}")
            
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
                
            return None
        except Exception as e:
            logger.error(f"오디오 추출 중 예외 발생: {e}")
            
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
                
            return None
        
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Whisper API를 사용하여 오디오를 텍스트로 변환합니다.
        
        Args:
            audio_path (str): 오디오 파일 경로
            
        Returns:
            str: 변환된 텍스트
        """
        if audio_path is None:
            logger.error("오디오 파일 경로가 None입니다.")
            return ""
            
        if not os.path.exists(audio_path):
            logger.error(f"오디오 파일을 찾을 수 없음: {audio_path}")
            return ""
            
        logger.info(f"오디오 텍스트 변환 시작: {audio_path}")
        
        try:
            with open(audio_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="ko"
                )
            transcript = response.text
            logger.info(f"텍스트 변환 완료: {len(transcript)} 자")
            logger.debug(f"변환된 텍스트: {transcript[:200]}...")
            return transcript
        except Exception as e:
            logger.error(f"텍스트 변환 실패: {e}")
            return ""
        
    def analyze_detections(self, detections: List[Dict[str, Any]], video_path: Path = None) -> Dict[str, Any]:
        """
        감지된 로고들을 분석하여 PPL 효과를 평가합니다.
        
        Args:
            detections (List[Dict[str, Any]]): YOLO로 감지된 로고 정보 리스트
            video_path (Path, optional): 음성 인식을 위한 원본 비디오 경로
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        logger.info("로고 감지 결과 분석 시작")
        logger.info(f"분석할 감지 결과 수: {len(detections)}")
        
        # 오디오 텍스트 추출 (비디오 경로가 제공된 경우)
        transcript = ""
        if video_path:
            try:
                audio_path = self.extract_audio(video_path)
                if audio_path:
                    transcript = self.transcribe_audio(audio_path)
                    # 임시 오디오 파일 삭제
                    try:
                        if os.path.exists(audio_path):
                            os.unlink(audio_path)
                            logger.info(f"임시 오디오 파일 삭제 완료: {audio_path}")
                    except Exception as e:
                        logger.error(f"임시 오디오 파일 삭제 실패: {e}")
                else:
                    logger.warning("오디오 추출에 실패하여 텍스트 맥락 없이 진행합니다.")
            except Exception as e:
                logger.error(f"오디오 처리 중 오류 발생: {e}")
                logger.warning("오디오 처리 오류로 인해 텍스트 맥락 없이 진행합니다.")
        
        # 로고별로 감지 정보 그룹화
        logo_groups = {}
        for detection in detections:
            logo_name = detection["class"]
            if logo_name not in logo_groups:
                logo_groups[logo_name] = []
            logo_groups[logo_name].append(detection)
            
        # 대략적인 영상 길이 계산
        video_duration = max(d["timestamp"] for d in detections) if detections else 0
        
        # 전체 대본을 시간대별로 나누기
        time_segmented_transcript = []
        if transcript and video_duration > 0:
            # 영상을 10개 정도의 구간으로 나누기
            segments = 10
            segment_duration = video_duration / segments
            transcript_chunks = self.chunk_transcript(transcript, segments)
            
            for i, chunk in enumerate(transcript_chunks):
                start_time = i * segment_duration
                end_time = (i + 1) * segment_duration
                time_segmented_transcript.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "content": chunk
                })
                
        # 각 로고별 분석 수행
        analysis_results = {}
        for logo_name, logo_detections in logo_groups.items():
            # 로고 노출 시간 계산
            total_duration = sum(d["timestamp"] for d in logo_detections)
            # 노출 빈도 계산
            frequency = len(logo_detections)
            # 평균 신뢰도 계산
            avg_confidence = sum(d["confidence"] for d in logo_detections) / len(logo_detections)
            
            # 로고 노출 시간대별 대화 내용 추출
            context_dialogues = []
            if transcript and time_segmented_transcript:
                for detection in logo_detections:
                    timestamp = detection["timestamp"]
                    # 해당 시간대의 대화 내용 찾기
                    for segment in time_segmented_transcript:
                        if segment["start_time"] <= timestamp < segment["end_time"]:
                            context_dialogues.append({
                                "timestamp": timestamp,
                                "dialogue": segment["content"]
                            })
                            break
            
            # LLM에 분석 요청
            prompt = self._create_analysis_prompt(
                logo_name, 
                total_duration, 
                frequency, 
                avg_confidence, 
                context_dialogues, 
                transcript, 
                time_segmented_transcript
            )
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "당신은 PPL 효과를 분석하는 전문가입니다."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                llm_analysis = response.choices[0].message.content
            except Exception as e:
                logger.error(f"LLM API 호출 실패: {e}")
                llm_analysis = f"LLM 분석 중 오류 발생: {str(e)}"
            
            analysis_results[logo_name] = {
                "total_duration": total_duration,
                "frequency": frequency,
                "avg_confidence": avg_confidence,
                "context_dialogues": context_dialogues if context_dialogues else None,
                "llm_analysis": llm_analysis
            }
        
        logger.info("로고 감지 결과 분석 완료")
        logger.debug(f"분석 결과: {analysis_results}")
        
        return analysis_results
    
    def _create_analysis_prompt(
        self, 
        logo_name: str, 
        total_duration: float, 
        frequency: int, 
        avg_confidence: float, 
        context_dialogues: List[Dict], 
        full_transcript: str,
        time_segmented_transcript: List[Dict]
    ) -> str:
        """
        LLM 분석을 위한 프롬프트를 생성합니다.
        
        Args:
            logo_name: 로고 이름
            total_duration: 총 노출 시간
            frequency: 노출 빈도
            avg_confidence: 평균 신뢰도
            context_dialogues: 로고 노출 시점의 대화 내용
            full_transcript: 전체 대본
            time_segmented_transcript: 시간대별로 나눈 대본
            
        Returns:
            str: 분석 프롬프트
        """
        prompt = f"""
        다음은 드라마/예능 프로그램에서 감지된 {logo_name} 브랜드의 PPL 정보입니다:
        
        # 기본 통계
        - 총 노출 시간: {total_duration:.2f}초
        - 노출 빈도: {frequency}회
        - 평균 신뢰도: {avg_confidence:.2f}
        """
        
        # 노출 시점별 대화 내용 추가
        if context_dialogues:
            prompt += f"\n\n# 브랜드 노출 시점의 대화/상황 맥락:"
            for i, ctx in enumerate(context_dialogues):
                if i >= 5:  # 너무 많은 경우 5개만 표시
                    prompt += f"\n\n(이하 {len(context_dialogues)-5}개 맥락 정보 생략...)"
                    break
                prompt += f"\n\n## 시점 {ctx['timestamp']:.2f}초:"
                prompt += f"\n{ctx['dialogue']}"
        
        # 전체 대본 요약 추가 (너무 길면 일부만)
        if full_transcript:
            summary_length = min(len(full_transcript), 1500)  # 너무 길면 잘라서 제공
            prompt += f"\n\n# 전체 대본 요약 (처음 일부):"
            prompt += f"\n{full_transcript[:summary_length]}"
            
            if len(full_transcript) > summary_length:
                prompt += "\n...(이하 생략)..."
        
        # 시간대별 대본 추가
        if time_segmented_transcript:
            prompt += f"\n\n# 시간대별 대본:"
            for segment in time_segmented_transcript:
                # 각 세그먼트마다 간단한 요약만 포함
                summary = segment["content"][:200] + ("..." if len(segment["content"]) > 200 else "")
                prompt += f"\n\n## {segment['start_time']:.2f}초 ~ {segment['end_time']:.2f}초:"
                prompt += f"\n{summary}"
        
        prompt += f"""
        
        # 분석 요청
        위 정보를 바탕으로 다음을 분석해주세요:
        
        1. PPL 효과성 (1-10점) 및 점수 선정 이유
        2. 주요 노출 시점과 특징 분석
        3. 대화/상황 맥락과 PPL의 연관성 (맥락 활용이 효과적인지)
        4. 시청자의 시선을 끌기 위한 전략이 있었는지 분석
        5. PPL의 자연스러움 및 개선 제안사항
        """
        
        return prompt.strip()
        
    def chunk_transcript(self, transcript: str, num_chunks: int) -> List[str]:
        """
        텍스트를 대략적으로 동일한 크기의 청크로 나눕니다.
        
        Args:
            transcript (str): 변환된 텍스트
            num_chunks (int): 원하는 청크 수
            
        Returns:
            List[str]: 청크 목록
        """
        # 빈 문자열이면 빈 리스트 반환
        if not transcript:
            return []
            
        # 문장을 기준으로 분리
        sentences = transcript.replace('. ', '.|').replace('! ', '!|').replace('? ', '?|').split('|')
        
        # 빈 문장 제거
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 문장이 없으면 빈 리스트 반환
        if not sentences:
            return []
            
        # 청크 수가 문장 수보다 많으면 문장 수로 조정
        num_chunks = min(num_chunks, len(sentences))
        
        # 청크 수가 0이면 빈 리스트 반환
        if num_chunks <= 0:
            return []
            
        # 각 청크에 포함될 문장 수 계산
        sentences_per_chunk = max(1, len(sentences) // num_chunks)
        
        # 청크 생성
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = ' '.join(sentences[i:i+sentences_per_chunk])
            chunks.append(chunk)
        
        return chunks 