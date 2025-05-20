import os
import sys
from dotenv import load_dotenv
from util.agent_state import AgentState
from workflows.agentic_rag_graph import AgenticRAGGraph
from datetime import datetime
import logging
import yaml
from typing import Dict, Any
from agents.research_collector import ResearchCollector
from agents.data_preprocessor import DataPreprocessor
from agents.data_retriever import DataRetriever
from agents.tech_trend_summarizer import TechTrendSummarizer
from agents.trend_prediction import TrendPredictor
from agents.report_writer import ReportWriter

# 재귀 깊이 제한 증가
sys.setrecursionlimit(10000)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_environment():
    """필요한 디렉토리와 환경 변수를 확인합니다."""
    # 필요한 디렉토리 생성
    os.makedirs("data/vectordb", exist_ok=True)
    
    # 환경 변수 로드
    load_dotenv()
    
    # 필수 환경 변수 확인
    required_vars = [
        "OPENAI_API_KEY",
        "TAVILY_API_KEY",
        "SERPAPI_API_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"필수 환경 변수가 없습니다: {', '.join(missing_vars)}")

def load_config() -> Dict[str, Any]:
    """설정 파일 로드"""
    try:
        with open("config/agent_prompts.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"설정 파일 로드 실패: {str(e)}")
        raise

def main():
    """메인 실행 함수"""
    try:
        # 환경 검증
        validate_environment()
        
        # 설정 로드
        config = load_config()
        
        # 초기 상태 생성
        state = AgentState(
            query="UAM 기술 동향 및 전망",
            time_range="2023-01-01",
            keywords=["UAM", "eVTOL", "도심항공모빌리티", "항공기술"]
        )
        
        # 에이전트 초기화
        research_collector = ResearchCollector()
        data_preprocessor = DataPreprocessor()
        data_retriever = DataRetriever()
        tech_trend_summarizer = TechTrendSummarizer()
        trend_predictor = TrendPredictor()
        report_writer = ReportWriter()
        
        # 워크플로우 실행
        logger.info("연구 데이터 수집 시작")
        state = research_collector.run(state)
        
        logger.info("데이터 전처리 시작")
        state = data_preprocessor.run(state)
        
        logger.info("데이터 검색 시작")
        state = data_retriever.run(state)
        
        logger.info("기술 트렌드 요약 시작")
        state = tech_trend_summarizer.run(state)
        
        logger.info("트렌드 예측 시작")
        state = trend_predictor.run(state)
        
        logger.info("보고서 작성 시작")
        state = report_writer.run(state)
        
        # 결과 출력
        if state.report:
            print("\n=== 최종 보고서 ===")
            print(state.report)
        else:
            print("\n보고서 생성 실패")
        
        # 에러 출력
        if state.errors:
            print("\n=== 발생한 에러 ===")
            for error in state.errors:
                print(f"- {error}")
        
    except Exception as e:
        logger.error(f"워크플로우 실행 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 