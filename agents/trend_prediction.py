import logging
import yaml
import os
import json
from typing import List, Dict, Any
from openai import OpenAI
from util.agent_state import AgentState, TrendPrediction
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class TrendPrediction:
    """기술 트렌드를 예측하는 에이전트"""
    
    def __init__(self):
        """초기화"""
        # .env 파일 로드
        load_dotenv()
        
        # API 키 로드
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error("OpenAI API 키가 설정되지 않았습니다.")
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
        
        self.client = OpenAI(api_key=self.openai_api_key)
        self.prompts = self._load_prompts()
        logger.info("트렌드 예측 에이전트 초기화 완료")
        
    def _load_prompts(self) -> Dict:
        """YAML 파일에서 프롬프트 로드"""
        try:
            with open("config/agent_prompts.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config["trend_prediction"]
        except Exception as e:
            logger.error(f"프롬프트 로드 실패: {str(e)}")
            raise
    
    def run(self, state: AgentState) -> AgentState:
        """트렌드 예측 실행"""
        try:
            if not state.trends:
                logger.warning("예측할 트렌드가 없습니다.")
                return state
            
            # 트렌드 예측
            predictions = self._predict_trends(state.trends)
            
            # 예측 결과를 상태에 추가
            for prediction in predictions:
                state.predictions.append(prediction)
            
            return state
            
        except Exception as e:
            error_msg = f"트렌드 예측 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            state.add_error(error_msg)
            return state
    
    def _predict_trends(self, trends: List[Any]) -> List[TrendPrediction]:
        """트렌드 예측"""
        try:
            if not trends:
                logger.warning("예측할 트렌드가 없습니다.")
                return []

            logger.info(f"총 {len(trends)}개의 트렌드 예측 시작")
            
            # 트렌드 정보를 텍스트로 변환
            text = "\n\n".join([
                f"트렌드: {trend.name}\n"
                f"설명: {trend.description}\n"
                f"중요도: {trend.importance}\n"
                f"증거: {', '.join(trend.evidence)}"
                for trend in trends
            ])
            
            logger.debug(f"예측할 텍스트 크기: {len(text)} 문자")
            
            # GPT 모델 호출
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": self.prompts['system_prompt']},
                    {"role": "user", "content": self.prompts['prompts']['prediction'].format(
                        trend_name=trends[0].name,  # 첫 번째 트렌드 정보로 프롬프트 포맷팅
                        trend_description=trends[0].description,
                        trend_evidence=', '.join(trends[0].evidence)
                    )}
                ],
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            logger.debug(f"GPT 응답 수신: {len(result)} 문자")
            logger.debug(f"GPT 응답 내용: {result[:500]}...")
            
            try:
                # JSON 문자열을 파싱
                prediction_data = json.loads(result)
                logger.debug(f"파싱된 예측 데이터 구조: {json.dumps(prediction_data, ensure_ascii=False, indent=2)[:1000]}...")
                
                if not isinstance(prediction_data, dict):
                    logger.error("응답이 딕셔너리 형식이 아닙니다")
                    return []
                
                # 예측 데이터 생성
                prediction = TrendPrediction(
                    prediction=str(prediction_data.get('prediction', '')),
                    confidence=float(prediction_data.get('confidence', 0.5)),
                    timeframe=str(prediction_data.get('timeframe', '')),
                    market_impact=str(prediction_data.get('market_impact', '')),
                    risks=list(prediction_data.get('risks', [])),
                    opportunities=list(prediction_data.get('opportunities', [])),
                    recommendations=list(prediction_data.get('recommendations', []))
                )
                
                logger.info(f"트렌드 '{trends[0].name}' 예측 완료")
                return [prediction]
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 실패: {str(e)}")
                logger.debug(f"원본 응답: {result}")
                return []
            
        except Exception as e:
            logger.error(f"트렌드 예측 중 오류 발생: {str(e)}")
            return [] 