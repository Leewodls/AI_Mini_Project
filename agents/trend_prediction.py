import logging
import yaml
import os
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
            # 트렌드 정보를 텍스트로 변환
            text = "\n\n".join([
                f"트렌드: {trend.name}\n설명: {trend.description}\n중요도: {trend.importance}"
                for trend in trends
            ])
            
            # GPT 모델 호출
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": self.prompts['system_prompt']},
                    {"role": "user", "content": self.prompts['prompts']['trend_forecasting'].format(
                        text=text
                    )}
                ],
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            predictions_data = result.get('predictions', [])
            
            return [
                TrendPrediction(
                    trend_name=pred['trend_name'],
                    prediction=pred['prediction'],
                    timeframe=pred['timeframe'],
                    confidence=pred['confidence'],
                    factors=pred['factors']
                )
                for pred in predictions_data
            ]
            
        except Exception as e:
            logger.error(f"트렌드 예측 중 오류 발생: {str(e)}")
            return [] 