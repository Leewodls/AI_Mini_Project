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
            
            # 트렌드 정보를 텍스트로 변환
            text_parts = []
            for trend in trends:
                try:
                    # evidence 처리
                    if isinstance(trend.evidence, (list, tuple)):
                        evidence_str = ', '.join(str(e) for e in trend.evidence)
                    else:
                        evidence_str = str(trend.evidence)
                    
                    # 텍스트 부분 생성
                    text_part = (
                        f"트렌드: {str(trend.name)}\n"
                        f"설명: {str(trend.description)}\n"
                        f"중요도: {str(trend.importance)}\n"
                        f"증거: {evidence_str}"
                    )
                    text_parts.append(text_part)
                except Exception as e:
                    logger.warning(f"트렌드 변환 중 오류 발생 (무시됨): {str(e)}")
                    continue
            
            if not text_parts:
                logger.error("변환된 텍스트가 없습니다.")
                return []
            
            text = "\n\n".join(text_parts)
            logger.debug(f"예측할 텍스트 크기: {len(text)} 문자")
            
            # GPT 모델 호출
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {"role": "system", "content": self.prompts.get('system_prompt', '')},
                        {"role": "user", "content": self.prompts.get('prompts', {}).get('prediction', '').format(
                            trend_name=trends[0].name,
                            trend_description=trends[0].description,
                            trend_evidence=', '.join(str(e) for e in trends[0].evidence) if isinstance(trends[0].evidence, (list, tuple)) else str(trends[0].evidence)
                        )}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                    max_tokens=1500
                )
            except Exception as e:
                logger.error(f"GPT API 호출 실패: {str(e)}")
                return []
            
            if not response or not response.choices:
                logger.error("GPT 응답이 비어있거나 잘못되었습니다.")
                logger.debug(f"원본 응답: {response}")
                return []
            
            result = response.choices[0].message.content
            if not result:
                logger.error("GPT 응답 내용이 비어있습니다.")
                return []
            
            logger.debug(f"GPT 응답 수신: {len(result)} 문자")
            logger.debug(f"GPT 응답 내용: {result[:500]}...")
            
            try:
                # JSON 문자열을 파싱
                prediction_data = json.loads(result)
                if not isinstance(prediction_data, dict):
                    logger.error("응답이 딕셔너리 형식이 아닙니다")
                    return []
                
                logger.debug(f"GPT 응답 데이터: {json.dumps(prediction_data, indent=2, ensure_ascii=False)}")

                # 예측 데이터 생성
                try:
                    trend_prediction = TrendPrediction(
                        prediction=str(prediction_data.get('prediction', '')),
                        confidence=float(prediction_data.get('confidence', 0.5)),
                        timeframe=str(prediction_data.get('timeframe', '')),
                        market_impact=str(prediction_data.get('market_impact', '')),
                        risks=[str(r) for r in prediction_data.get('risks', [])],
                        opportunities=[str(o) for o in prediction_data.get('opportunities', [])],
                        recommendations=[str(r) for r in prediction_data.get('recommendations', [])]
                    )
                    logger.info(f"예측 데이터 생성 완료: {trend_prediction.prediction[:50]}...")
                    logger.debug(f"생성된 예측 데이터: {trend_prediction}")
                except Exception as e:
                    logger.error(f"예측 데이터 생성 중 오류: {str(e)}")
                    logger.debug(f"예측 데이터 생성 시도 데이터: {prediction_data}")
                    return []
                    
                logger.info(f"트렌드 '{trends[0].name}' 예측 완료")
                return [trend_prediction]
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 실패: {str(e)}")
                logger.debug(f"원본 응답: {result}")
                return []
            
        except Exception as e:
            logger.error(f"트렌드 예측 중 오류 발생: {str(e)}")
            logger.debug(f"트렌드 리스트: {trends}")
            return []