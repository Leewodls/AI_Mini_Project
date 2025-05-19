from typing import Dict, Any, List
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer
from util.config import get_agent_config
from util.agent_state import AgentState, ResearchData, ProcessedData
import os
from tools.vector_store import VectorStore
import logging
import yaml
from openai import OpenAI
from dotenv import load_dotenv
import json
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """수집된 데이터를 전처리하는 에이전트"""
    
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
        self.timeout = 30  # API 호출 타임아웃 (초)
        logger.info("데이터 전처리 에이전트 초기화 완료")
        
    def _load_prompts(self) -> Dict:
        """YAML 파일에서 프롬프트 로드"""
        try:
            with open("config/agent_prompts.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config["data_preprocessor"]
        except Exception as e:
            logger.error(f"프롬프트 로드 실패: {str(e)}")
            raise
    
    async def _process_text_async(self, text: str, prompt_key: str) -> str:
        """텍스트 처리 (비동기)"""
        try:
            # 프롬프트 가져오기
            prompt = self.prompts['prompts'].get(prompt_key)
            if not prompt:
                logger.error(f"프롬프트를 찾을 수 없음: {prompt_key}")
                return None
            
            # 시스템 프롬프트 가져오기
            system_prompt = self.prompts.get('system_prompt', '')
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openai_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4-turbo-preview",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt.format(text=text)}
                        ],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.3  # 더 일관된 결과를 위해 낮은 온도 사용
                    },
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content']
                    else:
                        error_text = await response.text()
                        logger.error(f"OpenAI API 호출 실패: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"텍스트 처리 중 오류 발생: {str(e)}")
            return None
    
    async def _clean_text_async(self, text: str) -> str:
        """텍스트 정제 (비동기)"""
        if not text:
            return ""
            
        try:
            # GPT를 사용한 텍스트 정제
            response = await self._process_text_async(
                text,
                'text_cleaning'
            )
            
            if not response:
                logger.warning("텍스트 정제 실패")
                return text
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"텍스트 정제 중 오류 발생: {str(e)}")
            return text
    
    async def _extract_keywords_async(self, text: str) -> List[str]:
        """키워드 추출 (비동기)"""
        if not text:
            return []
            
        try:
            # GPT를 사용한 키워드 추출
            response = await self._process_text_async(
                text,
                'keyword_extraction'
            )
            
            if not response:
                logger.warning("키워드 추출 실패")
                # 기본 키워드 추출 시도
                words = text.split()
                # 2글자 이상 단어만 선택
                keywords = [w for w in words if len(w) >= 2]
                # 상위 5개 단어 반환
                return keywords[:5]
            
            try:
                # JSON 응답 파싱
                keywords_data = json.loads(response)
                
                # 다양한 키워드 필드 시도
                keywords = (
                    keywords_data.get('keywords', []) or
                    keywords_data.get('key_points', []) or
                    keywords_data.get('terms', []) or
                    []
                )
                
                # 키워드 정제
                keywords = [
                    k.strip() for k in keywords 
                    if k and isinstance(k, str) and len(k.strip()) >= 2
                ]
                
                # 중복 제거
                keywords = list(dict.fromkeys(keywords))
                
                # 최대 10개 키워드 반환
                return keywords[:10]
                
            except json.JSONDecodeError as e:
                logger.error(f"키워드 데이터 파싱 실패: {str(e)}")
                # 기본 키워드 추출 시도
                words = text.split()
                keywords = [w for w in words if len(w) >= 2]
                return keywords[:5]
            
        except Exception as e:
            logger.error(f"키워드 추출 중 오류 발생: {str(e)}")
            # 기본 키워드 추출 시도
            words = text.split()
            keywords = [w for w in words if len(w) >= 2]
            return keywords[:5]
    
    async def _chunk_text_async(self, text: str) -> List[Dict[str, Any]]:
        """텍스트 청크 분할 (비동기)"""
        result = await self._process_text_async(text, 'text_chunking')
        if result:
            try:
                data = json.loads(result)
                return data['chunks'] if 'chunks' in data else []
            except json.JSONDecodeError:
                return []
        return []
    
    async def _extract_metadata_async(self, research_data: Any) -> Dict[str, Any]:
        """메타데이터 추출 (비동기)"""
        result = await self._process_text_async(research_data.summary, 'metadata_extraction')
        if result:
            try:
                data = json.loads(result)
                metadata = dict(data)
                
                # 원본 데이터의 메타데이터 추가
                metadata.update({
                    'title': research_data.title,
                    'source': research_data.source,
                    'url': research_data.url,
                    'type': research_data.type
                })
                
                return metadata
            except json.JSONDecodeError:
                pass
        
        return {
            'title': research_data.title,
            'source': research_data.source,
            'url': research_data.url,
            'type': research_data.type
        }
    
    async def _process_research_data_async(self, research_data: ResearchData) -> ProcessedData:
        """연구 데이터 처리 (비동기)"""
        # 병렬로 처리
        cleaned_text, chunks, metadata = await asyncio.gather(
            self._clean_text_async(research_data.summary),
            self._chunk_text_async(research_data.summary),
            self._extract_metadata_async(research_data)
        )
        
        return ProcessedData(
            original_id=research_data.id,
            cleaned_text=cleaned_text,
            chunks=chunks,
            metadata=metadata
        )
    
    async def run_async(self, state: AgentState) -> AgentState:
        """비동기 데이터 전처리 실행"""
        try:
            if not state.research_data:
                logger.warning("전처리할 연구 데이터가 없습니다.")
                return state
            
            logger.info(f"총 {len(state.research_data)}개의 데이터 전처리 시작")
            
            # 각 데이터 전처리
            for i, data in enumerate(state.research_data):
                try:
                    # 텍스트 정제
                    cleaned_text = await self._clean_text_async(data.summary)
                    if not cleaned_text:
                        logger.warning(f"데이터 {i+1}의 텍스트 정제 실패")
                        cleaned_text = data.summary  # 원본 텍스트 사용
                    
                    # 키워드 추출
                    keywords = await self._extract_keywords_async(cleaned_text)
                    
                    # 데이터 업데이트
                    data.summary = cleaned_text
                    data.key_points = keywords
                    
                    logger.debug(f"데이터 {i+1} 전처리 완료: {len(keywords)}개 키워드")
                    
                except Exception as e:
                    logger.error(f"데이터 {i+1} 전처리 중 오류 발생: {str(e)}")
                    # 최소한의 데이터는 유지
                    data.key_points = data.summary.split()[:5] if data.summary else []
                    continue
            
            # 전처리된 데이터 필터링 (최소한의 데이터는 유지)
            state.research_data = [data for data in state.research_data if data.summary]
            
            logger.info(f"전처리 완료: {len(state.research_data)}개 데이터")
            return state
            
        except Exception as e:
            error_msg = f"데이터 전처리 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            state.add_error(error_msg)
            return state
    
    def run(self, state: AgentState) -> AgentState:
        """데이터 전처리 실행"""
        return asyncio.run(self.run_async(state)) 