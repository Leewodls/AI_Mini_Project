import logging
from typing import Dict, Any
from util.agent_state import AgentState
from agents.research_collector import ResearchCollector
from agents.data_preprocessor import DataPreprocessor
from agents.data_retriever import DataRetriever
from agents.tech_trend_summarizer import TechTrendSummarizer
from agents.trend_prediction import TrendPrediction
from agents.report_writer import ReportWriter

logger = logging.getLogger(__name__)

class AgenticRAGGraph:
    """에이전트 기반 RAG 워크플로우"""
    
    def __init__(self):
        """초기화"""
        self.research_collector = ResearchCollector()
        self.data_preprocessor = DataPreprocessor()
        self.data_retriever = DataRetriever()
        self.tech_trend_summarizer = TechTrendSummarizer()
        self.trend_prediction = TrendPrediction()
        self.report_writer = ReportWriter()
        logger.info("AgenticRAGGraph 초기화 완료")
    
    def run(self, state: AgentState) -> AgentState:
        """워크플로우 실행"""
        try:
            # 1. 연구 데이터 수집
            logger.info("연구 데이터 수집 시작")
            state = self.research_collector.run(state)
            
            # 2. 데이터 전처리
            logger.info("데이터 전처리 시작")
            state = self.data_preprocessor.run(state)
            
            # 3. 데이터 검색
            logger.info("데이터 검색 시작")
            state = self.data_retriever.run(state)
            
            # 4. 기술 트렌드 요약
            logger.info("기술 트렌드 요약 시작")
            state = self.tech_trend_summarizer.run(state)
            
            # 5. 트렌드 예측
            logger.info("트렌드 예측 시작")
            state = self.trend_prediction.run(state)
            
            # 6. 보고서 작성
            logger.info("보고서 작성 시작")
            state = self.report_writer.run(state)
            
            logger.info("워크플로우 실행 완료")
            return state
            
        except Exception as e:
            error_msg = f"워크플로우 실행 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            state.add_error(error_msg)
            return state