import logging
import yaml
import os
import json
from typing import List, Dict, Any
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from util.agent_state import AgentState, RetrievedData
from dotenv import load_dotenv
from tools.vector_store import VectorStore
import asyncio

logger = logging.getLogger(__name__)

class DataRetriever:
    """벡터 DB에서 관련 데이터를 검색하는 에이전트"""
    
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
        self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        self.vector_store = VectorStore()
        self.prompts = self._load_prompts()
        logger.info("데이터 검색 에이전트 초기화 완료")
    
    def _load_prompts(self) -> Dict:
        """YAML 파일에서 프롬프트 로드"""
        try:
            with open("config/agent_prompts.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config["data_retriever"]
        except Exception as e:
            logger.error(f"프롬프트 로드 실패: {str(e)}")
            raise
    
    def run(self, state: AgentState) -> AgentState:
        """데이터 검색 실행"""
        try:
            # 의미론적 검색 수행
            search_results = self._semantic_search(state.query)
            
            # 검색 결과를 상태에 추가
            for result in search_results:
                retrieved_data = RetrievedData(
                    title=result['title'],
                    content=result['content'],
                    relevance_score=result['relevance_score'],
                    key_points=result['key_points'],
                    source=result.get('source', '알 수 없음'),
                    url=result.get('url', ''),
                    type=result.get('type', 'research'),
                    metadata=result.get('metadata', {})
                )
                state.retrieved_data.append(retrieved_data)
                logger.debug(f"검색 결과 추가: {retrieved_data.title} (출처: {retrieved_data.source})")
            
            return state
            
        except Exception as e:
            error_msg = f"데이터 검색 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            state.add_error(error_msg)
            return state
    
    def _semantic_search(self, query: str) -> List[Dict[str, Any]]:
        """의미론적 검색 수행"""
        try:
            # 벡터 DB에서 검색
            search_results = self.vector_store.search(query)
            
            if not search_results:
                logger.warning(f"검색 결과가 없습니다: {query}")
                return []
            
            # 검색 결과 분석
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": self.prompts['system_prompt']},
                    {"role": "user", "content": self.prompts['prompts']['semantic_search'].format(
                        query=query,
                        results=search_results
                    )}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            try:
                result = json.loads(content)
                if isinstance(result, dict) and 'relevant_docs' in result:
                    # 검색 결과에 메타데이터 추가
                    for doc in result['relevant_docs']:
                        if isinstance(doc, dict):
                            metadata = doc.get('metadata', {})
                            doc['source'] = metadata.get('source', '알 수 없음')
                            doc['url'] = metadata.get('url', '')
                            doc['type'] = metadata.get('type', 'research')
                            doc['metadata'] = metadata
                    return result['relevant_docs']
                return search_results
            except json.JSONDecodeError:
                logger.error("JSON 파싱 실패")
                return search_results
            
        except Exception as e:
            logger.error(f"의미론적 검색 중 오류 발생: {str(e)}")
            return []
    
    async def _semantic_search_async(self, query: str) -> List[Dict[str, Any]]:
        """의미론적 검색 수행 (비동기)"""
        try:
            # Vector DB 검색
            results = await asyncio.to_thread(
                self.vector_store.search,
                query=query,
                n_results=10
            )
            
            if not results:
                logger.warning("검색 결과가 없습니다.")
                return []
            
            # 검색 결과 처리
            processed_results = []
            for result in results:
                if isinstance(result, dict):
                    metadata = result.get('metadata', {})
                    processed_results.append({
                        'title': result.get('title', ''),
                        'content': result.get('content', ''),
                        'relevance_score': result.get('relevance_score', 0.0),
                        'key_points': metadata.get('key_points', []),
                        'source': metadata.get('source', '알 수 없음'),
                        'url': metadata.get('url', ''),
                        'type': metadata.get('type', 'research'),
                        'metadata': metadata
                    })
            
            return processed_results
            
        except Exception as e:
            logger.error(f"의미론적 검색 중 오류 발생: {str(e)}")
            return []
    
    async def run_async(self, state: AgentState) -> AgentState:
        """비동기 데이터 검색 실행"""
        try:
            # 검색 쿼리 생성
            query = state.query
            
            # 의미론적 검색 수행
            search_results = await self._semantic_search_async(query)
            
            if not search_results:
                logger.warning("검색 결과가 없습니다.")
                return state
            
            # 검색 결과를 상태에 추가
            for result in search_results:
                if isinstance(result, dict):
                    retrieved_data = RetrievedData(
                        title=result.get('title', ''),
                        content=result.get('content', ''),
                        relevance_score=result.get('relevance_score', 0.0),
                        key_points=result.get('key_points', []),
                        source=result.get('source', '알 수 없음'),
                        url=result.get('url', ''),
                        type=result.get('type', 'research'),
                        metadata=result.get('metadata', {})
                    )
                    state.retrieved_data.append(retrieved_data)
                    logger.debug(f"검색 결과 추가: {retrieved_data.title} (출처: {retrieved_data.source})")
            
            return state
            
        except Exception as e:
            error_msg = f"데이터 검색 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            state.add_error(error_msg)
            return state