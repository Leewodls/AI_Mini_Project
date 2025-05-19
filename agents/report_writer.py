from typing import Dict, Any, List
from datetime import datetime
import openai
import os
from util.config import get_agent_config
from util.agent_state import AgentState, Trend, TrendPrediction, NewsData, Report
import logging
from openai import OpenAI
import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class ReportWriter:
    """보고서를 작성하는 에이전트"""
    
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
        logger.info("보고서 작성 에이전트 초기화 완료")
    
    def _load_prompts(self) -> Dict:
        """YAML 파일에서 프롬프트 로드"""
        try:
            with open("config/agent_prompts.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config["report_writer"]
        except Exception as e:
            logger.error(f"프롬프트 로드 실패: {str(e)}")
            raise
    
    def generate_summary(self, trends: List[Trend], predictions: List[TrendPrediction]) -> str:
        """
        LLM을 사용하여 주요 트렌드와 예측을 요약합니다.
        """
        try:
            if not self.openai_api_key:
                return self._generate_summary_fallback(trends, predictions)
            
            summary_text = f"트렌드: {[trend.name for trend in trends]}\n예측: {[pred.technology for pred in predictions]}"
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 UAM 기술 트렌드 분석 전문가입니다. 모든 결과는 반드시 한국어로 작성하세요."},
                    {"role": "user", "content": f"다음 트렌드와 예측 데이터를 바탕으로 요약을 작성해주세요. 반드시 한국어로 작성하세요.\n\n{summary_text}"}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM 요약 생성 중 오류: {str(e)}")
            return self._generate_summary_fallback(trends, predictions)
    
    def _generate_summary_fallback(self, trends: List[Trend], predictions: List[TrendPrediction]) -> str:
        """
        LLM 사용이 불가능할 때의 기본 요약 생성
        """
        summary = "주요 기술 트렌드 및 예측 요약:\n\n"
        
        summary += "주요 트렌드:\n"
        for trend in trends[:5]:
            summary += f"- {trend.name}: {trend.description}\n"
        
        summary += "\n주요 예측:\n"
        for pred in predictions[:5]:
            summary += f"- {pred.technology}: {pred.prediction}\n"
        
        return summary
    
    def format_trend_section(self, trends: List[Trend]) -> str:
        """
        LLM을 사용하여 트렌드 섹션을 포맷팅합니다.
        """
        try:
            if not self.openai_api_key:
                return self._format_trend_section_fallback(trends)
            
            # 트렌드 데이터 준비
            trends_text = "\n".join([
                f"트렌드명: {trend.name}\n설명: {trend.description}\n증거: {', '.join(trend.evidence)}\n관련 뉴스: {[news.title for news in trend.related_news]}"
                for trend in trends
            ])
            
            # LLM에 요청
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 UAM 기술 트렌드 분석 전문가입니다. 모든 결과는 반드시 한국어로 작성하세요."},
                    {"role": "user", "content": f"다음 트렌드 데이터를 바탕으로 트렌드 분석 보고서를 작성해주세요. 반드시 한국어로 작성하세요.\n\n{trends_text}"}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"LLM 트렌드 섹션 생성 중 오류: {str(e)}")
            return self._format_trend_section_fallback(trends)
    
    def _format_trend_section_fallback(self, trends: List[Trend]) -> str:
        """
        LLM 사용이 불가능할 때의 기본 트렌드 섹션 포맷팅
        """
        section = "2. 기술 트렌드 분석\n\n"
        
        for trend in trends:
            section += f"### {trend.name}\n"
            section += f"{trend.description}\n\n"
            
            if trend.evidence:
                section += "지원 증거:\n"
                for evidence in trend.evidence:
                    section += f"- {evidence}\n"
                section += "\n"
        
        return section
    
    def format_prediction_section(self, predictions: List[TrendPrediction]) -> str:
        """
        LLM을 사용하여 예측 섹션을 포맷팅합니다.
        """
        try:
            if not self.openai_api_key:
                return self._format_prediction_section_fallback(predictions)
            
            # 예측 데이터 준비
            predictions_text = "\n".join([
                f"기술: {pred.technology}\n예측: {pred.prediction}\n신뢰도: {pred.confidence}\n시기: {pred.timeframe}\n시장 적용성: {pred.market_applicability}\n위험 요소: {', '.join(pred.risk_factors)}\n시장 증거: {[news.title for news in pred.market_evidence]}"
                for pred in predictions
            ])
            
            # LLM에 요청
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 UAM 기술 트렌드 분석 전문가입니다. 모든 결과는 반드시 한국어로 작성하세요."},
                    {"role": "user", "content": f"다음 예측 데이터를 바탕으로 예측 보고서를 작성해주세요. 반드시 한국어로 작성하세요.\n\n{predictions_text}"}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"LLM 예측 섹션 생성 중 오류: {str(e)}")
            return self._format_prediction_section_fallback(predictions)
    
    def _format_prediction_section_fallback(self, predictions: List[TrendPrediction]) -> str:
        """
        LLM 사용이 불가능할 때의 기본 예측 섹션 포맷팅
        """
        section = "3. 기술 발전 예측\n\n"
        
        for pred in predictions:
            section += f"### {pred.technology}\n"
            section += f"예측: {pred.prediction}\n"
            section += f"신뢰도: {pred.confidence:.2f}\n"
            section += f"예상 시기: {pred.timeframe}\n"
            section += f"시장 적용성: {pred.market_applicability:.2f}\n\n"
            
            if pred.risk_factors:
                section += "위험 요소:\n"
                for risk in pred.risk_factors:
                    section += f"- {risk}\n"
                section += "\n"
        
        return section
    
    def generate_conclusion(self, trends: List[Trend], predictions: List[TrendPrediction]) -> str:
        """
        LLM을 사용하여 결론을 생성합니다.
        """
        try:
            if not self.openai_api_key:
                return self._generate_conclusion_fallback(trends, predictions)
            
            conclusion_text = f"트렌드: {[trend.name for trend in trends]}\n예측: {[pred.technology for pred in predictions]}"
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 UAM 기술 트렌드 분석 전문가입니다. 모든 결과는 반드시 한국어로 작성하세요."},
                    {"role": "user", "content": f"다음 트렌드와 예측 데이터를 바탕으로 결론을 작성해주세요. 반드시 한국어로 작성하세요.\n\n{conclusion_text}"}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM 결론 생성 중 오류: {str(e)}")
            return self._generate_conclusion_fallback(trends, predictions)
    
    def _generate_conclusion_fallback(self, trends: List[Trend], predictions: List[TrendPrediction]) -> str:
        """
        LLM 사용이 불가능할 때의 기본 결론 생성
        """
        return "본 보고서는 UAM 기술의 현재 트렌드와 미래 발전 방향을 분석했습니다. 기술적 발전과 시장 동향을 종합적으로 고려하여, UAM 기술의 상용화와 확산에 대한 전망을 제시했습니다."
    
    def generate_report(self, state: AgentState) -> str:
        """
        전체 보고서를 생성합니다.
        """
        report = f"# UAM 기술 트렌드 분석 보고서\n"
        report += f"생성일: {datetime.now().strftime('%Y-%m-%d')}\n\n"
        
        # 요약
        report += self.generate_summary(state.trends, state.predictions)
        report += "\n"
        
        # 트렌드 분석
        report += self.format_trend_section(state.trends)
        report += "\n"
        
        # 예측
        report += self.format_prediction_section(state.predictions)
        report += "\n"
        
        # 결론
        report += "4. 결론\n"
        report += self.generate_conclusion(state.trends, state.predictions)
        
        return report
    
    def run(self, state: AgentState) -> AgentState:
        """보고서 작성 실행"""
        try:
            if not state.trends or not state.predictions:
                logger.warning("보고서 작성에 필요한 데이터가 없습니다.")
                return state
            
            # 보고서 생성
            report = self._generate_report(state.trends, state.predictions)
            
            # 생성된 보고서를 상태에 추가
            state.report = report
            
            logger.info("보고서 작성이 완료되었습니다.")
            return state
            
        except Exception as e:
            error_msg = f"보고서 작성 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            state.add_error(error_msg)
            return state
    
    def _generate_report(self, trends: List[Any], predictions: List[Any]) -> Report:
        """보고서 생성"""
        try:
            # 트렌드와 예측 정보를 텍스트로 변환
            trends_text = "\n\n".join([
                f"트렌드: {trend.name}\n설명: {trend.description}\n중요도: {trend.importance}"
                for trend in trends
            ])
            
            predictions_text = "\n\n".join([
                f"트렌드: {pred.trend_name}\n예측: {pred.prediction}\n기간: {pred.timeframe}\n신뢰도: {pred.confidence}"
                for pred in predictions
            ])
            
            # GPT 모델 호출
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": self.prompts['system_prompt']},
                    {"role": "user", "content": self.prompts['prompts']['report_generation'].format(
                        trends=trends_text,
                        predictions=predictions_text
                    )}
                ],
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            report_data = result.get('report', {})
            
            return Report(
                title=report_data['title'],
                summary=report_data['summary'],
                trends=report_data['trends'],
                predictions=report_data['predictions'],
                recommendations=report_data['recommendations']
            )
            
        except Exception as e:
            logger.error(f"보고서 생성 중 오류 발생: {str(e)}")
            return None 