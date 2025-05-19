import logging
import yaml
import os
from typing import List, Dict, Any
from openai import OpenAI
from util.agent_state import AgentState, ResearchData
from tavily import TavilyClient
from serpapi import GoogleSearch
from dotenv import load_dotenv
import json
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

class ResearchCollector:
    """연구 데이터를 수집하는 에이전트"""
    
    def __init__(self):
        """초기화"""
        # .env 파일 로드
        load_dotenv()
        
        # API 키 로드 및 검증
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.serpapi_api_key = os.getenv("SERPAPI_API_KEY")
        
        if not self.openai_api_key:
            logger.error("OpenAI API 키가 설정되지 않았습니다.")
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
        
        if not self.tavily_api_key:
            logger.warning("Tavily API 키가 설정되지 않았습니다. SerpAPI만 사용합니다.")
        
        if not self.serpapi_api_key:
            logger.warning("SerpAPI 키가 설정되지 않았습니다. Tavily API만 사용합니다.")
        
        if not any([self.tavily_api_key, self.serpapi_api_key]):
            logger.error("검색 API 키가 하나도 설정되지 않았습니다.")
            raise ValueError("검색 API 키가 필요합니다.")
        
        self.client = OpenAI(api_key=self.openai_api_key)
        if self.tavily_api_key:
            self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
        self.prompts = self._load_prompts()
        self.timeout = 30  # API 호출 타임아웃 (초)
        logger.info("연구 데이터 수집 에이전트 초기화 완료")
        
    def _load_prompts(self) -> Dict:
        """YAML 파일에서 프롬프트 로드"""
        try:
            with open("config/agent_prompts.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config["research_collector"]
        except Exception as e:
            logger.error(f"프롬프트 로드 실패: {str(e)}")
            raise
    
    async def _search_tavily_academic_async(self, query: str) -> List[ResearchData]:
        """Tavily API를 사용한 학술 검색 (비동기)"""
        if not self.tavily_api_key:
            logger.warning("Tavily API 키가 없어 학술 검색을 건너뜁니다.")
            return []
            
        try:
            logger.info(f"Tavily 학술 검색 시작: {query}")
            
            # TavilyClient를 사용한 검색
            search_results = await asyncio.to_thread(
                self.tavily_client.search,
                query=query,
                search_depth="advanced",
                include_answer=True,
                search_type="academic"
            )
            
            logger.debug(f"Tavily 학술 검색 응답: {json.dumps(search_results, ensure_ascii=False, indent=2)}")
            
            if not search_results or 'results' not in search_results:
                logger.warning(f"Tavily 학술 검색 결과가 없습니다: {query}")
                return []
            
            results = [
                ResearchData(
                    id=f"academic_{i}",
                    title=result.get('title', ''),
                    source=result.get('authors', ''),
                    summary=result.get('abstract', ''),
                    url=result.get('url', ''),
                    type='academic'
                )
                for i, result in enumerate(search_results['results'])
            ]
            
            logger.info(f"Tavily 학술 검색 완료: {len(results)}개 결과")
            return results
            
        except Exception as e:
            logger.error(f"Tavily 학술 검색 중 오류 발생: {str(e)}")
            return []
    
    async def _search_tavily_news_async(self, query: str) -> List[ResearchData]:
        """Tavily API를 사용한 뉴스 검색 (비동기)"""
        if not self.tavily_api_key:
            logger.warning("Tavily API 키가 없어 뉴스 검색을 건너뜁니다.")
            return []
            
        try:
            logger.info(f"Tavily 뉴스 검색 시작: {query}")
            
            # TavilyClient를 사용한 검색
            search_results = await asyncio.to_thread(
                self.tavily_client.search,
                query=query,
                search_depth="advanced",
                include_answer=True,
                search_type="news"
            )
            
            logger.debug(f"Tavily 뉴스 검색 응답: {json.dumps(search_results, ensure_ascii=False, indent=2)}")
            
            if not search_results or 'results' not in search_results:
                logger.warning(f"Tavily 뉴스 검색 결과가 없습니다: {query}")
                return []
            
            results = [
                ResearchData(
                    id=f"news_{i}",
                    title=result.get('title', ''),
                    source=result.get('source', ''),
                    summary=result.get('summary', ''),
                    url=result.get('url', ''),
                    type='news'
                )
                for i, result in enumerate(search_results['results'])
            ]
            
            logger.info(f"Tavily 뉴스 검색 완료: {len(results)}개 결과")
            return results
            
        except Exception as e:
            logger.error(f"Tavily 뉴스 검색 중 오류 발생: {str(e)}")
            return []
    
    def _search_serpapi(self, query: str) -> List[ResearchData]:
        """SerpAPI를 사용한 검색"""
        if not self.serpapi_api_key:
            logger.warning("SerpAPI 키가 없어 웹 검색을 건너뜁니다.")
            return []
            
        try:
            logger.info(f"SerpAPI 검색 시작: {query}")
            
            # SerpAPI 검색 파라미터 설정
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.serpapi_api_key,
                "num": 10,  # 검색 결과 수
                "gl": "kr",  # 한국 지역 설정
                "hl": "ko",  # 한국어 결과
                "google_domain": "google.co.kr"  # 한국 구글 도메인 사용
            }
            
            logger.debug(f"SerpAPI 파라미터: {params}")
            
            # 검색 실행
            search = GoogleSearch(params)
            results = search.get_dict()
            
            logger.debug(f"SerpAPI 응답: {json.dumps(results, ensure_ascii=False, indent=2)}")
            
            # 검색 결과 처리
            if 'organic_results' in results:
                search_results = []
                for i, result in enumerate(results['organic_results']):
                    try:
                        search_results.append(
                            ResearchData(
                                id=f"web_{i}",
                                title=result.get('title', ''),
                                source=result.get('source', ''),
                                summary=result.get('snippet', ''),
                                url=result.get('link', ''),
                                type='web'
                            )
                        )
                    except Exception as e:
                        logger.error(f"결과 처리 중 오류 발생 (인덱스 {i}): {str(e)}")
                        continue
                
                logger.info(f"SerpAPI 검색 완료: {len(search_results)}개 결과")
                return search_results
                
            elif 'error' in results:
                logger.error(f"SerpAPI 검색 오류: {results['error']}")
                return []
            else:
                logger.warning("SerpAPI 검색 결과가 없습니다.")
                return []
            
        except Exception as e:
            logger.error(f"SerpAPI 검색 중 오류 발생: {str(e)}")
            return []
    
    async def run_async(self, state: AgentState) -> AgentState:
        """비동기 데이터 수집 실행"""
        try:
            logger.info(f"연구 데이터 수집 시작: {state.query}")
            
            # 병렬로 검색 실행
            tasks = []
            
            # SerpAPI 검색 (동기)
            serpapi_results = self._search_serpapi(state.query)
            state.research_data.extend(serpapi_results)
            
            # Tavily 검색 (비동기)
            if self.tavily_api_key:
                logger.info("Tavily 검색 시작")
                academic_task = asyncio.create_task(self._search_tavily_academic_async(state.query))
                news_task = asyncio.create_task(self._search_tavily_news_async(state.query))
                tasks.extend([academic_task, news_task])
            
            # 모든 검색 결과 수집
            if tasks:
                logger.info(f"총 {len(tasks)}개의 검색 작업 실행 중...")
                tavily_results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, results in enumerate(tavily_results):
                    if isinstance(results, Exception):
                        logger.error(f"Tavily 검색 {i+1} 중 오류 발생: {str(results)}")
                        continue
                    state.research_data.extend(results)
            
            # 결과 정리
            if not state.research_data:
                logger.warning("검색 결과가 없습니다.")
            else:
                # 중복 제거
                unique_results = {}
                for result in state.research_data:
                    if result.url not in unique_results:
                        unique_results[result.url] = result
                
                state.research_data = list(unique_results.values())
                logger.info(f"총 {len(state.research_data)}개의 고유한 검색 결과가 수집되었습니다.")
            
            return state
            
        except Exception as e:
            error_msg = f"연구 데이터 수집 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            state.add_error(error_msg)
            return state
    
    def run(self, state: AgentState) -> AgentState:
        """데이터 수집 실행"""
        return asyncio.run(self.run_async(state)) 