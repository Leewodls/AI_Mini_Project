from typing import Dict, Any
import yaml
import os
import logging

logger = logging.getLogger(__name__)

class PromptTemplates:
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Any]:
        """
        프롬프트 템플릿을 로드합니다.
        """
        try:
            template_path = os.path.join("config", "agent_prompts.yaml")
            if os.path.exists(template_path):
                with open(template_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"프롬프트 템플릿 파일을 찾을 수 없습니다: {template_path}")
                return self._get_default_templates()
        except Exception as e:
            logger.error(f"프롬프트 템플릿 로드 중 오류 발생: {str(e)}")
            return self._get_default_templates()
    
    def _get_default_templates(self) -> Dict[str, Any]:
        """
        기본 프롬프트 템플릿을 반환합니다.
        """
        return {
            "research_collector": {
                "system": "당신은 UAM 기술 트렌드 분석을 위한 연구 데이터 수집 전문가입니다. 주어진 쿼리에 대해 관련성 높은 연구 논문과 기사를 수집해주세요.",
                "user": "다음 쿼리에 대한 연구 데이터를 수집해주세요: {query}\n시간 범위: {time_range}\n키워드: {keywords}"
            },
            "data_preprocessor": {
                "system": "당신은 UAM 기술 트렌드 분석을 위한 데이터 전처리 전문가입니다. 수집된 데이터를 정제하고 구조화해주세요.",
                "user": "다음 데이터를 전처리해주세요:\n{data}"
            },
            "data_retriever": {
                "system": "당신은 UAM 기술 트렌드 분석을 위한 데이터 검색 전문가입니다. 주어진 쿼리와 관련된 문서를 검색해주세요.",
                "user": "다음 쿼리와 관련된 문서를 검색해주세요: {query}\n검색 결과 수: {n_results}"
            },
            "tech_trend_summarizer": {
                "system": "당신은 UAM 기술 트렌드 분석 전문가입니다. 주어진 데이터를 바탕으로 주요 기술 트렌드를 추출하고 요약해주세요.",
                "user": "다음 데이터를 바탕으로 기술 트렌드를 분석해주세요:\n{data}"
            },
            "trend_prediction": {
                "system": "당신은 UAM 기술 트렌드 예측 전문가입니다. 현재 트렌드를 바탕으로 미래 기술 발전을 예측해주세요.",
                "user": "다음 트렌드를 바탕으로 미래 예측을 해주세요:\n{trends}"
            },
            "report_writer": {
                "system": "당신은 UAM 기술 트렌드 분석 보고서 작성 전문가입니다. 분석 결과를 바탕으로 체계적인 보고서를 작성해주세요. 모든 결과는 반드시 한국어로 작성하세요.",
                "user": "다음 분석 결과를 바탕으로 보고서를 작성해주세요. 반드시 한국어로 작성하세요:\n트렌드: {trends}\n예측: {predictions}"
            }
        }
    
    def get_prompt(self, agent_name: str, **kwargs) -> Dict[str, str]:
        """
        에이전트의 프롬프트를 반환합니다.
        """
        try:
            template = self.templates.get(agent_name, {})
            if not template:
                logger.warning(f"에이전트 '{agent_name}'의 프롬프트 템플릿을 찾을 수 없습니다.")
                return {}
            
            return {
                "system": template["system"],
                "user": template["user"].format(**kwargs)
            }
        except Exception as e:
            logger.error(f"프롬프트 생성 중 오류 발생: {str(e)}")
            return {} 