# AI_Mini_Project

'''
UAM_Tech_Trend_Analysis/
├── .env                    # 환경 변수 파일
├── main.py                 # 메인 실행 파일
├── requirements.txt        # 의존성 패키지 목록
├── agents/                 # 에이전트 모듈들
│   ├── __init__.py
│   ├── research_collector.py      # 연구 및 뉴스 수집 에이전트
│   ├── data_preprocessor.py       # 데이터 전처리 에이전트
│   ├── data_retriever.py          # 데이터 검색 에이전트
│   ├── tech_trend_summarizer.py   # 기술 트렌드 요약 에이전트
│   ├── trend_prediction.py        # 기술 발전 예측 에이전트
│   └── report_writer.py           # 보고서 작성 에이전트
├── data/                   # 데이터 저장소
│   ├── vectordb/              # 벡터 DB 저장소
│   └── raw/                   # 수집된 원본 데이터
├── tools/                  # 유틸리티 도구들
│   ├── __init__.py
│   ├── hybrid_retriever.py    # 하이브리드 검색 도구
│   └── prompt_templates.py    # 프롬프트 템플릿 관리
├── util/                   # 유틸리티 함수들
│   ├── __init__.py
│   ├── config.py              # 환경 설정 관리
│   └── agent_state.py         # 에이전트 상태 관리
├── workflows/              # 워크플로우 정의
│   ├── __init__.py
│   └── agentic_rag_graph.py   # LangGraph 기반 에이전트 그래프
├── outputs/                # 분석 결과 저장
│   └── reports/              # 보고서 파일 저장
└── config/                 # 설정 파일
    └── uam_agent_prompts.yaml   # 에이전트 프롬프트 설정 파일
'''

## 환경 설정

프로젝트를 실행하기 전에 다음 환경 변수를 설정해야 합니다:            

1. `.env` 파일을 프로젝트 루트 디렉토리에 생성하고 다음 변수들을 설정하세요:

```bash
# API 키 설정
SERP_API_KEY=your_serp_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
YOUTUBE_API_KEY=your_youtube_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# 데이터베이스 설정
VECTOR_DB_PATH=data/vectordb
RAW_DATA_PATH=data/raw

# 출력 설정
OUTPUT_DIR=outputs/reports

# 기타 설정
LOG_LEVEL=INFO
MAX_TOKENS=512
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## 설치 및 실행

1. 가상환경 생성 및 활성화:
```bash
python -m venv my_env
source my_env/bin/activate  # Linux/Mac
# 또는
my_env\Scripts\activate  # Windows
```

2. 의존성 패키지 설치:
```bash
pip install -r requirements.txt
```

3. 프로그램 실행:
```bash
python main.py
```

## 주요 기능

1. 연구 데이터 수집
   - Google Scholar API를 통한 논문 수집
   - SerpAPI를 통한 뉴스 기사 수집
   - YouTube API를 통한 관련 영상 수집

2. 데이터 전처리
   - 텍스트 정제 및 청크 분할
   - 임베딩 생성 및 벡터 DB 저장

3. 기술 트렌드 분석
   - 주요 기술 트렌드 추출
   - 시장 적용 가능성 평가
   - 리스크 요인 식별

4. 보고서 생성
   - 트렌드 요약
   - 예측 분석
   - 시장 전망
   - 리스크 분석
