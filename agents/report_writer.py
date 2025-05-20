from typing import Dict, Any, List, Optional
from datetime import datetime
import openai
import os
from util.config import get_agent_config
from util.agent_state import AgentState, Trend, TrendPrediction, NewsData, Report
import logging
from openai import OpenAI
import yaml
from dotenv import load_dotenv
import json

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
            
            summary_text = f"트렌드: {[trend.name for trend in trends]}\n예측: {[pred.prediction for pred in predictions]}"
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "당신은 UAM 기술 트렌드 분석 전문가입니다. 모든 결과는 반드시 한국어로 작성하세요."},
                    {"role": "user", "content": f"다음 트렌드와 예측 데이터를 바탕으로 요약을 작성해주세요. 반드시 한국어로 작성하세요.\n\n{summary_text}"}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM 요약 생성 중 오류: {str(e)}")
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
            summary += f"- {pred.prediction} (신뢰도: {pred.confidence:.2f})\n"
        
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
                f"예측: {pred.prediction}\n"
                f"신뢰도: {pred.confidence}\n"
                f"시기: {pred.timeframe}\n"
                f"시장 영향: {pred.market_impact}\n"
                f"위험 요소: {', '.join(pred.risks)}\n"
                f"기회 요소: {', '.join(pred.opportunities)}\n"
                f"권장사항: {', '.join(pred.recommendations)}"
                for pred in predictions
            ])
            
            # LLM에 요청
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "당신은 UAM 기술 트렌드 분석 전문가입니다. 모든 결과는 반드시 한국어로 작성하세요."},
                    {"role": "user", "content": f"다음 예측 데이터를 바탕으로 예측 보고서를 작성해주세요. 반드시 한국어로 작성하세요.\n\n{predictions_text}"}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM 예측 섹션 생성 중 오류: {str(e)}")
            return self._format_prediction_section_fallback(predictions)
    
    def _format_prediction_section_fallback(self, predictions: List[TrendPrediction]) -> str:
        """
        LLM 사용이 불가능할 때의 기본 예측 섹션 포맷팅
        """
        section = "3. 기술 발전 예측\n\n"
        
        for pred in predictions:
            section += f"### 예측: {pred.prediction}\n"
            section += f"신뢰도: {pred.confidence:.2f}\n"
            section += f"예상 시기: {pred.timeframe}\n"
            section += f"시장 영향: {pred.market_impact}\n\n"
            
            if pred.risks:
                section += "위험 요소:\n"
                for risk in pred.risks:
                    section += f"- {risk}\n"
                section += "\n"
            
            if pred.opportunities:
                section += "기회 요소:\n"
                for opp in pred.opportunities:
                    section += f"- {opp}\n"
                section += "\n"
            
            if pred.recommendations:
                section += "권장사항:\n"
                for rec in pred.recommendations:
                    section += f"- {rec}\n"
                section += "\n"
        
        return section
    
    def generate_conclusion(self, trends: List[Trend], predictions: List[TrendPrediction]) -> str:
        """
        LLM을 사용하여 결론을 생성합니다.
        """
        try:
            if not self.openai_api_key:
                return self._generate_conclusion_fallback(trends, predictions)
            
            conclusion_text = f"트렌드: {[trend.name for trend in trends]}\n예측: {[pred.prediction for pred in predictions]}"
            
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
            report_obj = self._generate_report(state.trends, state.predictions)
            if report_obj is None:
                logger.error("보고서 생성에 실패했습니다.")
                state.add_error("보고서 생성 실패")
                return state
            
            # Report 객체를 마크다운 형식의 문자열로 변환
            report_md = self._convert_report_to_markdown(report_obj)
            state.report = report_md
            
            logger.info("보고서 작성이 완료되었습니다.")
            return state
            
        except Exception as e:
            error_msg = f"보고서 작성 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            state.add_error(error_msg)
            return state
    
    def _convert_report_to_markdown(self, report: Report) -> str:
        """Report 객체를 마크다운 형식의 문자열로 변환"""
        md = f"# {report.title}\n\n"
        md += f"## 실행 요약\n{report.executive_summary}\n\n"
        md += f"## 서론\n{report.introduction}\n\n"
        md += f"## 방법론\n{report.methodology}\n\n"
        md += f"## 주요 발견사항\n{report.findings}\n\n"
        md += f"## 트렌드 분석\n{report.trend_analysis}\n\n"
        md += f"## 미래 전망\n{report.future_outlook}\n\n"
        md += f"## 결론\n{report.conclusion}\n\n"
        md += f"## 권장사항\n{report.recommendations}\n\n"
        if report.appendix:
            md += f"## 부록\n{report.appendix}\n"
        return md
    
    def _generate_report(self, trends: List[Trend], predictions: List[TrendPrediction]) -> Optional[Report]:
        """보고서 생성"""
        try:
            # 트렌드와 예측 정보를 텍스트로 변환
            trends_text = "\n\n".join([
                f"트렌드: {trend.name}\n"
                f"설명: {trend.description}\n"
                f"중요도: {trend.importance}\n"
                f"증거: {', '.join(trend.evidence)}"
                for trend in trends
            ])
            
            predictions_text = "\n\n".join([
                f"예측: {pred.prediction}\n"
                f"시기: {pred.timeframe}\n"
                f"신뢰도: {pred.confidence}\n"
                f"시장 영향: {pred.market_impact}\n"
                f"위험 요소: {', '.join(pred.risks)}\n"
                f"기회 요소: {', '.join(pred.opportunities)}\n"
                f"권장사항: {', '.join(pred.recommendations)}"
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
            try:
                report_data = json.loads(result)
                if not isinstance(report_data, dict):
                    raise ValueError("응답이 딕셔너리 형식이 아닙니다")
                
                # 필수 필드 검증
                required_fields = ['title', 'executive_summary', 'introduction', 'methodology', 
                                 'findings', 'trend_analysis', 'future_outlook', 'conclusion', 
                                 'recommendations']
                for field in required_fields:
                    if field not in report_data or not report_data[field]:
                        logger.warning(f"필수 필드 '{field}'가 누락되었거나 비어있습니다.")
                        report_data[field] = f"[{field} 내용 없음]"
                
                return Report(
                    title=report_data['title'],
                    executive_summary=report_data['executive_summary'],
                    introduction=report_data['introduction'],
                    methodology=report_data['methodology'],
                    findings=report_data['findings'],
                    trend_analysis=report_data['trend_analysis'],
                    future_outlook=report_data['future_outlook'],
                    conclusion=report_data['conclusion'],
                    recommendations=report_data['recommendations'],
                    appendix=report_data.get('appendix', '')
                )
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 실패: {str(e)}")
                return None
            
        except Exception as e:
            logger.error(f"보고서 생성 중 오류 발생: {str(e)}")
            return None 