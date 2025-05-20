import logging
import yaml
import os
import json
from typing import List, Dict, Any
from openai import OpenAI
from util.agent_state import AgentState, TrendPrediction, Trend
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class TrendPredictor:
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
            
            # 트렌드 데이터 유효성 검사
            valid_trends = []
            for trend in state.trends:
                if not isinstance(trend, Trend):
                    logger.warning(f"잘못된 트렌드 객체 형식: {trend}")
                    continue
                if not trend.name or not trend.description:
                    logger.warning(f"필수 필드가 비어있는 트렌드 발견: {trend}")
                    continue
                valid_trends.append(trend)
            
            if not valid_trends:
                logger.error("유효한 트렌드가 없습니다.")
                state.add_error("유효한 트렌드가 없습니다.")
                return state
            
            # 트렌드 예측
            predictions = self._predict_trends(valid_trends)
            
            # 예측 결과를 상태에 추가
            logger.info(f"생성된 예측 수: {len(predictions)}")
            for prediction in predictions:
                if isinstance(prediction, TrendPrediction):
                    state.predictions.append(prediction)
                    logger.debug(f"예측 추가됨: {prediction.prediction[:50]}...")
                else:
                    logger.warning(f"잘못된 예측 객체 형식: {prediction}")
            
            logger.info(f"최종 상태의 예측 수: {len(state.predictions)}")
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

            logger.info(f"트렌드 예측 시작: {len(trends)}개의 트렌드")
            predictions = []
            
            # 각 트렌드에 대해 예측 생성
            for trend in trends:
                try:
                    # evidence 처리
                    if isinstance(trend.evidence, (list, tuple)):
                        evidence_str = ', '.join(str(e) for e in trend.evidence)
                    else:
                        evidence_str = str(trend.evidence)
                    
                    # GPT 모델 호출
                    response = self.client.chat.completions.create(
                        model="gpt-4-turbo-preview",
                        messages=[
                            {"role": "system", "content": self.prompts.get('system_prompt', '')},
                            {"role": "user", "content": self.prompts.get('prompts', {}).get('prediction', '').format(
                                trend_name=trend.name,
                                trend_description=trend.description,
                                trend_evidence=evidence_str
                            )}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.3,
                        max_tokens=1500
                    )
                    
                    if not response or not response.choices:
                        logger.error(f"트렌드 '{trend.name}'에 대한 GPT 응답이 비어있거나 잘못되었습니다.")
                        continue
                    
                    result = response.choices[0].message.content
                    if not result:
                        logger.error(f"트렌드 '{trend.name}'에 대한 GPT 응답 내용이 비어있습니다.")
                        continue
                    
                    # JSON 파싱
                    prediction_data = json.loads(result)
                    if not isinstance(prediction_data, dict):
                        logger.error(f"트렌드 '{trend.name}'에 대한 응답이 딕셔너리 형식이 아닙니다")
                        continue
                    
                    # 예측 데이터 생성
                    trend_prediction = TrendPrediction(
                        prediction=str(prediction_data.get('prediction', '')),
                        confidence=float(prediction_data.get('confidence', 0.5)),
                        timeframe=str(prediction_data.get('timeframe', '')),
                        market_impact=str(prediction_data.get('market_impact', '')),
                        risks=[str(r) for r in prediction_data.get('risks', [])],
                        opportunities=[str(o) for o in prediction_data.get('opportunities', [])],
                        recommendations=[str(r) for r in prediction_data.get('recommendations', [])]
                    )
                    
                    predictions.append(trend_prediction)
                    logger.info(f"트렌드 '{trend.name}' 예측 완료")
                    logger.debug(f"생성된 예측 데이터: {trend_prediction}")
                    
                except Exception as e:
                    logger.error(f"트렌드 '{trend.name}' 예측 중 오류 발생: {str(e)}")
                    continue
            
            logger.info(f"총 {len(predictions)}개의 예측 생성 완료")
            return predictions
            
        except Exception as e:
            logger.error(f"트렌드 예측 중 오류 발생: {str(e)}")
            logger.debug(f"트렌드 리스트: {trends}")
            return []