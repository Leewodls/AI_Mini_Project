import logging
import yaml
import os
from typing import List, Dict, Any
from openai import OpenAI
from util.agent_state import AgentState, Trend
from dotenv import load_dotenv
import json
import asyncio

logger = logging.getLogger(__name__)

class TechTrendSummarizer:
    """기술 트렌드를 요약하는 에이전트"""
    
    def __init__(self):
        """초기화"""
        # tokenizers 병렬 처리 설정
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # .env 파일 로드
        load_dotenv()
        
        # API 키 로드
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error("OpenAI API 키가 설정되지 않았습니다.")
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
        
        self.client = OpenAI(api_key=self.openai_api_key)
        self.prompts = self._load_prompts()
        logger.info("기술 트렌드 요약 에이전트 초기화 완료")
        
    def _load_prompts(self) -> Dict:
        """YAML 파일에서 프롬프트 로드"""
        try:
            with open("config/agent_prompts.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config["tech_trend_summarizer"]
        except Exception as e:
            logger.error(f"프롬프트 로드 실패: {str(e)}")
            raise
    
    def run(self, state: AgentState) -> AgentState:
        """기술 트렌드 요약 실행"""
        try:
            if not state.retrieved_data:
                logger.warning("요약할 검색 데이터가 없습니다.")
                return state
            
            # 데이터 유효성 검사
            valid_data = []
            for data in state.retrieved_data:
                if not hasattr(data, 'title') or not hasattr(data, 'content') or not hasattr(data, 'key_points'):
                    logger.warning(f"잘못된 데이터 형식 발견: {data}")
                    continue
                if not data.title or not data.content:
                    logger.warning(f"필수 필드가 비어있는 데이터 발견: {data}")
                    continue
                valid_data.append(data)
            
            if not valid_data:
                logger.error("유효한 검색 데이터가 없습니다.")
                state.add_error("유효한 검색 데이터가 없습니다.")
                return state
            
            # 검색된 데이터를 바탕으로 트렌드 요약
            trends = self._summarize_trends(valid_data)
            
            # 요약된 트렌드를 상태에 추가
            logger.info(f"생성된 트렌드 수: {len(trends)}")
            for trend in trends:
                if isinstance(trend, Trend):
                    state.trends.append(trend)
                    logger.debug(f"트렌드 추가됨: {trend.name}")
                else:
                    logger.warning(f"잘못된 트렌드 객체 형식: {trend}")
            
            logger.info(f"최종 상태의 트렌드 수: {len(state.trends)}")
            return state
            
        except Exception as e:
            error_msg = f"기술 트렌드 요약 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            state.add_error(error_msg)
            return state
    
    def _summarize_trends(self, retrieved_data: List[Any]) -> List[Trend]:
        """검색된 데이터를 바탕으로 트렌드 요약"""
        try:
            if not retrieved_data:
                logger.warning("요약할 검색 데이터가 없습니다.")
                return []

            logger.info(f"총 {len(retrieved_data)}개의 데이터로 트렌드 요약 시작")
            
            # 데이터 구조 로깅
            for idx, data in enumerate(retrieved_data, 1):
                logger.debug(f"데이터 {idx} 구조: title={type(data.title)}, content={type(data.content)}, source={data.source}")
            
            # 검색된 데이터를 텍스트로 변환
            text_parts = []
            for data in retrieved_data:
                try:
                    # key_points 처리
                    if isinstance(data.key_points, (list, tuple)):
                        key_points_str = ', '.join(str(point) for point in data.key_points)
                    else:
                        key_points_str = str(data.key_points)
                    
                    # 출처 정보 포함
                    source_info = f"출처: {data.source}"
                    if data.url:
                        source_info += f" ({data.url})"
                    
                    # 텍스트 부분 생성
                    text_part = f"제목: {str(data.title)}\n내용: {str(data.content)}\n주요 포인트: {key_points_str}\n{source_info}"
                    text_parts.append(text_part)
                except Exception as e:
                    logger.warning(f"데이터 변환 중 오류 발생 (무시됨): {str(e)}")
                    continue
            
            if not text_parts:
                logger.error("변환된 텍스트가 없습니다.")
                return []
            
            text = "\n\n".join(text_parts)
            logger.debug(f"요약할 텍스트 크기: {len(text)} 문자")
            
            # GPT 모델 호출
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": self.prompts['system_prompt']},
                    {"role": "user", "content": self.prompts['prompts']['trend_summarization'].format(
                        text=text
                    )}
                ],
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            
            try:
                # JSON 문자열을 파싱
                trends_data = json.loads(result)
                if not isinstance(trends_data, dict):
                    logger.error("응답이 딕셔너리 형식이 아닙니다")
                    return []
                
                # key_trends 필드 확인
                trends = trends_data.get('key_trends', [])
                if not isinstance(trends, list):
                    logger.error("트렌드 데이터가 리스트 형식이 아닙니다")
                    return []
                
                if not trends:
                    logger.warning("요약된 트렌드가 없습니다")
                    return []
                
                logger.info(f"총 {len(trends)}개의 트렌드 요약 완료")
                
                trend_objects = []
                for idx, trend in enumerate(trends, 1):
                    try:
                        if not isinstance(trend, dict):
                            logger.warning(f"트렌드 {idx}가 딕셔너리 형식이 아닙니다")
                            continue
                            
                        # 필수 필드 검증
                        name = trend.get('name')
                        description = trend.get('description')
                        evidence = trend.get('evidence', [])
                        importance = trend.get('importance', '중간')
                        sources = trend.get('sources', [])  # 출처 정보 추가
                        
                        if not name or not description:
                            logger.warning(f"트렌드 {idx}의 필수 정보가 누락되었습니다")
                            continue
                        
                        # importance 값 검증
                        if importance not in ['높음', '중간', '낮음']:
                            logger.warning(f"트렌드 {idx}의 중요도가 유효하지 않습니다: {importance}")
                            importance = '중간'
                        
                        # evidence 처리
                        if isinstance(evidence, (list, tuple)):
                            evidence_list = [str(e) for e in evidence]
                        else:
                            evidence_list = [str(evidence)]
                        
                        # 출처 정보 처리
                        if isinstance(sources, (list, tuple)):
                            sources_list = [str(s) for s in sources]
                        else:
                            sources_list = [str(sources)] if sources else []
                            
                        trend_obj = Trend(
                            name=str(name),
                            description=str(description),
                            evidence=evidence_list,
                            importance=importance,
                            sources=sources_list  # 출처 정보 추가
                        )
                        trend_objects.append(trend_obj)
                        logger.debug(f"트렌드 {idx}/{len(trends)} 생성 완료: {trend_obj.name} (출처: {', '.join(trend_obj.sources)})")
                        
                    except Exception as e:
                        logger.error(f"트렌드 {idx} 생성 중 오류 발생: {str(e)}")
                        continue
                
                return trend_objects
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 실패: {str(e)}")
                return []
            
        except Exception as e:
            logger.error(f"트렌드 요약 중 오류 발생: {str(e)}")
            return []

    async def _process_text_async(self, text: str, prompt_type: str) -> str:
        """GPT를 사용한 텍스트 처리 (비동기)"""
        try:
            # 프롬프트 가져오기
            prompt = self.prompts['prompts'].get(prompt_type)
            if not prompt:
                logger.error(f"프롬프트를 찾을 수 없음: {prompt_type}")
                return None
            
            # 시스템 프롬프트 가져오기
            system_prompt = self.prompts.get('system_prompt', '')
            
            # 프롬프트 포맷팅
            formatted_prompt = prompt.format(text=text)
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": formatted_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"텍스트 처리 중 오류 발생: {str(e)}")
            return None
    
    async def run_async(self, state: AgentState) -> AgentState:
        """비동기 트렌드 요약 실행"""
        try:
            if not state.research_data:
                logger.warning("요약할 연구 데이터가 없습니다.")
                return state
            
            logger.info(f"총 {len(state.research_data)}개의 데이터 요약 시작")
            
            # 연구 데이터를 텍스트로 변환
            research_text = "\n\n".join([
                f"제목: {data.title}\n"
                f"출처: {data.source}\n"
                f"요약: {data.summary}\n"
                f"키워드: {', '.join(data.key_points)}\n"
                f"URL: {data.url}"
                for data in state.research_data
            ])
            
            logger.debug(f"벡터 DB 저장 전 연구 데이터 크기: {len(research_text)} 문자")
            
            # GPT를 사용한 트렌드 요약
            response = await self._process_text_async(
                research_text,
                'trend_summary'
            )
            
            if not response:
                logger.error("트렌드 요약 생성 실패")
                return state
            
            try:
                # JSON 응답 파싱
                if isinstance(response, str):
                    summary_data = json.loads(response)
                elif isinstance(response, dict):
                    summary_data = response
                else:
                    logger.error(f"예상치 못한 응답 형식: {type(response)}")
                    return state
                
                # 응답 구조 검증
                if not isinstance(summary_data, dict):
                    logger.error("응답이 딕셔너리 형식이 아닙니다")
                    return state
                
                # key_trends 필드 검증
                key_trends = summary_data.get('key_trends', [])
                if not isinstance(key_trends, list):
                    logger.error("key_trends가 리스트 형식이 아닙니다")
                    return state
                
                logger.info(f"벡터 DB에 저장할 트렌드 수: {len(key_trends)}개")
                
                # 요약 데이터 저장
                state.trend_summary = {
                    'key_trends': key_trends,
                    'market_analysis': summary_data.get('market_analysis', {}),
                    'tech_challenges': summary_data.get('tech_challenges', []),
                    'future_outlook': summary_data.get('future_outlook', {})
                }
                
                # 트렌드 객체 생성 및 벡터 DB 저장 상태 로깅
                stored_trends = 0
                for idx, trend in enumerate(key_trends, 1):
                    if not isinstance(trend, dict):
                        logger.warning(f"잘못된 트렌드 형식: {trend}")
                        continue
                        
                    try:
                        trend_obj = Trend(
                            name=str(trend.get('name', '')),
                            description=str(trend.get('description', '')),
                            evidence=list(trend.get('evidence', [])),
                            importance=str(trend.get('importance', 'medium'))
                        )
                        state.trends.append(trend_obj)
                        stored_trends += 1
                        logger.debug(f"트렌드 {idx}/{len(key_trends)} 저장 완료: {trend_obj.name}")
                    except Exception as e:
                        logger.error(f"트렌드 {idx} 객체 생성 실패: {str(e)}")
                        continue
                
                logger.info(f"벡터 DB 저장 완료: 총 {stored_trends}/{len(key_trends)}개 트렌드 저장됨")
                logger.debug(f"저장된 트렌드 목록: {[t.name for t in state.trends]}")
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 실패: {str(e)}")
                logger.debug(f"원본 응답: {response}")
                state.add_error("JSON 파싱 실패")
            except Exception as e:
                logger.error(f"응답 처리 중 오류 발생: {str(e)}")
                logger.debug(f"원본 응답: {response}")
                state.add_error("응답 처리 실패")
            
            return state
            
        except Exception as e:
            error_msg = f"트렌드 요약 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            state.add_error(error_msg)
            return state 