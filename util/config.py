import os
import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """설정 파일 로드"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "agent_prompts.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"설정 파일 로드 중 오류 발생: {str(e)}")
        return {}

def get_agent_config(agent_name: str) -> Dict[str, Any]:
    """에이전트별 설정 반환"""
    try:
        config = load_config()
        
        # 에이전트 프롬프트 설정 반환
        if agent_name in config:
            return config[agent_name]
            
        # 에이전트 목록에서 찾기
        for agent in config.get("agents", []):
            if agent["name"].lower().replace(" ", "_") == agent_name.lower():
                return agent["prompt"]
        
        logger.warning(f"에이전트 '{agent_name}'의 설정을 찾을 수 없습니다.")
        return {}
        
    except Exception as e:
        logger.error(f"에이전트 설정 로드 중 오류 발생: {str(e)}")
        return {}

def get_vector_db_config() -> Dict[str, Any]:
    """Vector DB 설정 반환"""
    try:
        config = load_config()
        return config.get("settings", {}).get("vector_db", {})
    except Exception as e:
        logger.error(f"Vector DB 설정 로드 중 오류 발생: {str(e)}")
        return {}

def get_data_processing_config() -> Dict[str, Any]:
    """데이터 처리 설정 반환"""
    try:
        config = load_config()
        return config.get("settings", {}).get("data_processing", {})
    except Exception as e:
        logger.error(f"데이터 처리 설정 로드 중 오류 발생: {str(e)}")
        return {}

def get_output_config() -> Dict[str, Any]:
    """출력 설정 반환"""
    try:
        config = load_config()
        return config.get("settings", {}).get("output", {})
    except Exception as e:
        logger.error(f"출력 설정 로드 중 오류 발생: {str(e)}")
        return {} 