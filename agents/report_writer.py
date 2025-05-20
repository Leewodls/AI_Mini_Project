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
import pathlib
import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration
import re

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
        
        # 보고서 저장 디렉토리 설정
        self.report_dir = pathlib.Path("outputs/reports")
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # PDF 스타일 설정
        self.pdf_style = """
        @page {
            margin: 2.5cm;
            @top-right {
                content: counter(page);
            }
        }
        body {
            font-family: 'Noto Sans KR', sans-serif;
            line-height: 1.6;
            color: #333;
        }
        h1 {
            font-size: 24pt;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.3em;
            margin-top: 2em;
        }
        h2 {
            font-size: 20pt;
            color: #34495e;
            margin-top: 1.5em;
        }
        h3 {
            font-size: 16pt;
            color: #2c3e50;
            margin-top: 1.2em;
        }
        p {
            margin: 1em 0;
            text-align: justify;
        }
        ul, ol {
            margin: 1em 0;
            padding-left: 2em;
        }
        li {
            margin: 0.5em 0;
        }
        .metadata {
            font-size: 10pt;
            color: #7f8c8d;
            border-top: 1px solid #bdc3c7;
            margin-top: 2em;
            padding-top: 1em;
        }
        """
        
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
    
    def _save_report(self, report_md: str, report_obj: Report) -> str:
        """보고서를 파일로 저장 (PDF 형식)"""
        try:
            # 파일명 생성 (날짜_시간_제목)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = "".join(c for c in report_obj.title if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_title = safe_title.replace(' ', '_')
            base_filename = f"{timestamp}_{safe_title}"
            
            # 메타데이터 생성
            metadata = {
                "title": report_obj.title,
                "generated_at": datetime.now().isoformat(),
                "sections": {
                    "executive_summary": len(report_obj.executive_summary),
                    "introduction": len(report_obj.introduction),
                    "findings": len(report_obj.findings),
                    "trend_analysis": len(report_obj.trend_analysis),
                    "future_outlook": len(report_obj.future_outlook),
                    "conclusion": len(report_obj.conclusion),
                    "recommendations": len(report_obj.recommendations),
                    "appendix": len(report_obj.appendix)
                }
            }
            
            # 메타데이터를 YAML 형식으로 변환
            metadata_yaml = yaml.dump(metadata, allow_unicode=True, sort_keys=False)
            
            # 최종 보고서 내용 생성 (메타데이터 + 본문)
            final_content = f"---\n{metadata_yaml}---\n\n{report_md}"
            
            # 마크다운을 HTML로 변환
            html_content = markdown.markdown(
                final_content,
                extensions=['tables', 'fenced_code', 'codehilite']
            )
            
            # HTML 문서 생성
            html_doc = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>{report_obj.title}</title>
                <style>{self.pdf_style}</style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """
            
            # PDF 파일 경로
            pdf_path = self.report_dir / f"{base_filename}.pdf"
            
            # PDF 생성
            font_config = FontConfiguration()
            HTML(string=html_doc).write_pdf(
                pdf_path,
                stylesheets=[CSS(string=self.pdf_style)],
                font_config=font_config
            )
            
            logger.info(f"PDF 보고서가 저장되었습니다: {pdf_path}")
            return str(pdf_path)
            
        except Exception as e:
            logger.error(f"PDF 보고서 저장 중 오류 발생: {str(e)}")
            return None
    
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
            
            # 보고서 저장
            saved_path = self._save_report(report_md, report_obj)
            if saved_path:
                state.report = report_md
                state.report_path = saved_path
                logger.info(f"보고서가 성공적으로 저장되었습니다: {saved_path}")
            else:
                logger.error("보고서 저장에 실패했습니다.")
                state.add_error("보고서 저장 실패")
            
            return state
            
        except Exception as e:
            error_msg = f"보고서 작성 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            state.add_error(error_msg)
            return state
    
    def _convert_report_to_markdown(self, report: Report) -> str:
        """Report 객체를 Markdown 문자열로 변환"""
        markdown_content = f"# {report.title}\n\n"

        # Executive Summary
        if report.executive_summary:
            markdown_content += f"## Executive Summary\n\n{report.executive_summary}\n\n---\n\n"

        # Sections
        sections_to_include = [
            ("Introduction", report.introduction),
            ("Findings", report.findings),
            ("Trend Analysis", report.trend_analysis),
            ("Future Outlook", report.future_outlook),
            ("Conclusion", report.conclusion),
            ("Recommendations", report.recommendations),
            ("Appendix", report.appendix)
        ]

        for title, content in sections_to_include:
            if content:
                markdown_content += f"## {title}\n\n{content}\n\n"

        return markdown_content

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
            
            # 참고문헌 및 출처 정보 생성
            references = self._generate_references_content(trends, predictions)
            
            # 각 섹션별로 개별 생성
            sections = {}
            
            # 1. 제목 생성
            title_response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "당신은 UAM 기술 트렌드 분석 보고서 작성 전문가입니다. 주어진 정보를 바탕으로 보고서 제목을 생성해주세요. 제목만 생성하고 다른 설명은 하지 마세요."},
                    {"role": "user", "content": f"다음 정보를 바탕으로 보고서 제목을 생성해주세요:\n\n트렌드:\n{trends_text}\n\n예측:\n{predictions_text}"}
                ],
                temperature=0.7,
                max_tokens=100
            )
            sections['title'] = title_response.choices[0].message.content.strip()
            
            # 2. 실행 요약 생성
            summary_response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "당신은 UAM 기술 트렌드 분석 보고서 작성 전문가입니다. 주어진 정보를 바탕으로 5줄 이내의 실행 요약을 생성해주세요."},
                    {"role": "user", "content": f"다음 정보를 바탕으로 실행 요약을 생성해주세요:\n\n트렌드:\n{trends_text}\n\n예측:\n{predictions_text}"}
                ],
                temperature=0.7,
                max_tokens=500
            )
            sections['executive_summary'] = summary_response.choices[0].message.content.strip()
            
            # 3. 나머지 섹션 생성
            section_prompts = {
                'introduction': "UAM의 정의, 현재 시장 상황, 연구 목적을 포함한 서론을 작성해주세요.",
                'findings': "핵심 트렌드를 요약한 주요 발견사항을 작성해주세요.",
                'trend_analysis': "각 트렌드별 상세 분석을 작성해주세요.",
                'future_outlook': "예측 결과와 시나리오를 포함한 미래 전망을 작성해주세요.",
                'conclusion': "주요 시사점을 포함한 결론을 작성해주세요.",
                'recommendations': "실무적 제안을 포함한 권장사항을 작성해주세요."
            }
            
            for section, prompt in section_prompts.items():
                response = self.client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {"role": "system", "content": f"당신은 UAM 기술 트렌드 분석 보고서 작성 전문가입니다. {prompt}"},
                        {"role": "user", "content": f"다음 정보를 바탕으로 {section}를 작성해주세요:\n\n트렌드:\n{trends_text}\n\n예측:\n{predictions_text}"}
                    ],
                    temperature=0.7,
                    max_tokens=2300
                )
                sections[section] = response.choices[0].message.content.strip()
            
            # appendix는 참고문헌으로 설정
            sections['appendix'] = references
            
            # Report 객체 생성
            return Report(
                title=sections['title'],
                executive_summary=sections['executive_summary'],
                introduction=sections['introduction'],
                findings=sections['findings'],
                trend_analysis=sections['trend_analysis'],
                future_outlook=sections['future_outlook'],
                conclusion=sections['conclusion'],
                recommendations=sections['recommendations'],
                appendix=sections['appendix']
            )
            
        except Exception as e:
            logger.error(f"보고서 생성 중 오류 발생: {str(e)}")
            return None

    def _generate_references_content(self, trends: List[Trend], predictions: List[TrendPrediction]) -> str:
        """참고문헌 및 출처 내용 생성"""
        try:
            references_text = "### 핵심 키워드\n\n"
            
            # 트렌드에서 참고문헌 수집
            for trend in trends:
                if trend.evidence:
                    for evidence in trend.evidence:
                        references_text += f"- {evidence}\n"
            
            # 예측에서 참고문헌 수집
            for pred in predictions:
                if hasattr(pred, 'sources') and pred.sources:
                    for source in pred.sources:
                        references_text += f"- {source}\n"
            
            # 데이터 출처 추가
            references_text += "\n### 데이터 출처\n\n"
            references_text += "1. UAM 관련 뉴스 데이터베이스\n"
            references_text += "2. 기술 트렌드 분석 보고서\n"
            references_text += "3. 시장 조사 보고서\n"
            references_text += "4. 전문가 인터뷰 및 의견\n"
            references_text += "5. 산업 동향 보고서\n\n"
            
            # 생성 정보 추가
            references_text += "### 보고서 생성 정보\n\n"
            references_text += f"- 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            references_text += "- 생성 도구: GPT-4 기반 UAM 기술 트렌드 분석 시스템\n"
            references_text += "- 데이터 수집 기간: 최근 1년\n"
            
            return references_text
            
        except Exception as e:
            logger.error(f"참고문헌 내용 생성 중 오류 발생: {str(e)}")
            return "[참고문헌 내용 생성 실패]" 