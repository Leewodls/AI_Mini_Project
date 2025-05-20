from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import copy
import logging

logger = logging.getLogger(__name__)

@dataclass
class ResearchData:
    """연구 데이터 클래스"""
    id: str
    title: str
    source: str
    summary: str
    url: str
    type: str  # 'academic', 'news', 'web'

@dataclass
class NewsData:
    """뉴스 데이터 구조"""
    title: str
    content: str
    source: str
    date: str
    url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessedData:
    """전처리된 데이터 클래스"""
    original_id: str
    cleaned_text: str
    chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]

@dataclass
class RetrievedData:
    """검색된 데이터 클래스"""
    title: str
    content: str
    relevance_score: float
    key_points: List[str]
    source: str  # 출처 추가
    url: Optional[str] = None  # URL 추가
    type: str = 'research'  # 데이터 타입 추가
    metadata: Dict[str, Any] = field(default_factory=dict)  # 추가 메타데이터

@dataclass
class Trend:
    """기술 트렌드 클래스"""
    name: str
    description: str
    evidence: List[str]
    importance: str  # '높음', '중간', '낮음'
    sources: List[str] = field(default_factory=list)  # 출처 정보 추가
    related_news: List[NewsData] = field(default_factory=list)

@dataclass
class TrendPrediction:
    """트렌드 예측 클래스"""
    prediction: str = ''
    confidence: float = 0.5
    timeframe: str = ''
    market_impact: str = ''
    risks: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class Report:
    """분석 보고서 클래스"""
    title: str = ''
    executive_summary: str = ''
    introduction: str = ''
    findings: str = ''
    trend_analysis: str = ''
    future_outlook: str = ''
    conclusion: str = ''
    recommendations: str = ''
    appendix: str = ''

@dataclass
class AgentState:
    """에이전트 상태 클래스"""
    query: str
    time_range: str
    keywords: List[str]
    research_data: List[ResearchData] = field(default_factory=list)
    processed_data: List[ProcessedData] = field(default_factory=list)
    retrieved_data: List[RetrievedData] = field(default_factory=list)
    trends: List[Trend] = field(default_factory=list)
    predictions: List[TrendPrediction] = field(default_factory=list)
    report: Optional[Report] = None
    report_path: str = ""
    errors: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    logs: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def add_to_summary(self, key: str, value: Any) -> None:
        """요약 정보를 추가합니다."""
        self.summary[key] = value
        logger.debug(f"요약 정보 추가됨: {key}={value}")
    
    def get_state_summary(self) -> Dict[str, Any]:
        """상태 요약 반환"""
        return {
            "query": self.query,
            "time_range": self.time_range,
            "keywords": self.keywords,
            "research_data_count": len(self.research_data),
            "processed_data_count": len(self.processed_data),
            "retrieved_data_count": len(self.retrieved_data),
            "trends_count": len(self.trends),
            "predictions_count": len(self.predictions),
            "has_report": bool(self.report),
            "error_count": len(self.errors),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    def add_research_data(self, data: ResearchData) -> None:
        """연구 데이터를 추가합니다."""
        self.research_data.append(copy.deepcopy(data))
        logger.debug(f"연구 데이터 추가됨: {data.title}")
    
    def add_news_data(self, data: NewsData) -> None:
        """뉴스 데이터를 추가합니다."""
        self.news_data.append(copy.deepcopy(data))
    
    def add_processed_data(self, data: ProcessedData) -> None:
        """전처리된 데이터를 추가합니다."""
        self.processed_data.append(copy.deepcopy(data))
        logger.debug(f"전처리 데이터 추가됨: {data.cleaned_text[:100]}...")
    
    def add_retrieved_data(self, data: RetrievedData) -> None:
        """검색된 데이터를 추가합니다."""
        self.retrieved_data.append(copy.deepcopy(data))
        logger.debug(f"검색 데이터 추가됨: {data.title[:100]}...")
    
    def add_trend(self, trend: Trend) -> None:
        """기술 트렌드를 추가합니다."""
        self.trends.append(copy.deepcopy(trend))
        logger.debug(f"기술 트렌드 추가됨: {trend.name}")
    
    def add_prediction(self, prediction: TrendPrediction) -> None:
        """기술 예측을 추가합니다."""
        self.predictions.append(copy.deepcopy(prediction))
        logger.debug(f"예측 결과 추가됨: {prediction.prediction}")
    
    def update_report_content(self, content: str) -> None:
        """보고서 내용을 업데이트합니다."""
        self.report = content
    
    def add_error(self, error: str) -> None:
        """에러를 추가합니다."""
        self.errors.append(error)
        logger.error(f"에러 추가됨: {error}")
        self.updated_at = datetime.now()
    
    def add_log(self, log: str) -> None:
        """로그를 추가합니다."""
        self.logs.append(log)
        logger.info(f"로그 추가됨: {log}")
    
    def clear_errors(self) -> None:
        """에러를 초기화합니다."""
        self.errors.clear()
        self.updated_at = datetime.now()
    
    def validate_state(self) -> bool:
        """현재 상태가 유효한지 검증합니다."""
        try:
            # 필수 필드 검증
            if not self.query:
                raise ValueError("쿼리가 없습니다.")
            if not self.time_range:
                raise ValueError("시간 범위가 없습니다.")
            if not self.keywords:
                raise ValueError("키워드가 없습니다.")
            
            # 데이터 타입 검증
            if not isinstance(self.keywords, list):
                raise TypeError("keywords는 리스트여야 합니다.")
            
            # 데이터 일관성 검증
            if self.processed_data and not self.research_data:
                raise ValueError("전처리된 데이터가 있지만 원본 데이터가 없습니다.")
            
            if self.retrieved_documents and not self.processed_data:
                raise ValueError("검색된 데이터가 있지만 전처리된 데이터가 없습니다.")
            
            return True
            
        except Exception as e:
            self.errors.append(str(e))
            return False
            
    def validate_env_vars(self) -> bool:
        """필요한 환경 변수가 설정되어 있는지 검증합니다."""
        required_vars = [
            "SERP_API_KEY",
            "TAVILY_API_KEY",
            "YOUTUBE_API_KEY",
            "OPENAI_API_KEY"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            self.errors.append(f"필수 환경 변수가 없습니다: {', '.join(missing_vars)}")
            return False
            
        return True 