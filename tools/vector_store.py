import os
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging
from util.config import get_vector_db_config

logger = logging.getLogger(__name__)

class VectorStore:
    """벡터 데이터베이스 관리 클래스"""
    
    def __init__(self):
        """초기화"""
        self.config = get_vector_db_config()
        self.collection_name = self.config.get("collection_name", "uam_tech_trends")
        self.embedding_model = self.config.get("embedding_model", "paraphrase-multilingual-mpnet-base-v2")
        self.db_path = os.path.join(os.path.dirname(__file__), "..", "data", "vectordb")
        
        try:
            # 임베딩 모델 초기화
            self.model = SentenceTransformer(self.embedding_model)
            logger.info(f"임베딩 모델 로드 완료: {self.embedding_model}")
            
            # ChromaDB 클라이언트 초기화
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 컬렉션 생성 또는 로드
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Vector DB 컬렉션 '{self.collection_name}' 로드 완료")
            
        except Exception as e:
            logger.error(f"Vector DB 초기화 중 오류 발생: {str(e)}")
            raise
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        """메타데이터를 ChromaDB 호환 형식으로 변환"""
        formatted = {}
        for key, value in metadata.items():
            if value is None:
                formatted[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                formatted[key] = str(value)
            elif isinstance(value, (list, tuple)):
                formatted[key] = ", ".join(str(v) for v in value)
            elif isinstance(value, dict):
                formatted[key] = str(value)
            else:
                formatted[key] = str(value)
        return formatted
    
    def store(self, text: str, metadata: Dict[str, Any] = None) -> None:
        """텍스트를 벡터로 변환하여 저장"""
        try:
            # 텍스트 임베딩 생성
            embedding = self.model.encode(text).tolist()
            
            # 문서 ID 생성
            doc_id = f"doc_{len(self.collection.get()['ids'])}"
            
            # 메타데이터 포맷팅
            formatted_metadata = self._format_metadata(metadata or {})
            
            # Vector DB에 저장
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[formatted_metadata],
                documents=[text]
            )
            
            logger.info(f"텍스트 저장 완료: {text[:100]}...")
            
        except Exception as e:
            logger.error(f"텍스트 저장 중 오류 발생: {str(e)}")
            raise
    
    def search(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """쿼리와 유사한 텍스트 검색"""
        try:
            # 컬렉션 상태 확인
            collection_data = self.collection.get()
            if not collection_data['ids']:
                logger.warning("Vector DB가 비어있습니다. 기본 데이터를 추가합니다.")
                self._add_default_data()
                collection_data = self.collection.get()
            
            logger.info(f"현재 Vector DB 문서 수: {len(collection_data['ids'])}")
            
            # 쿼리 임베딩 생성
            query_embedding = self.model.encode(query).tolist()
            logger.debug(f"쿼리 임베딩 생성 완료: {query[:50]}...")
            
            # 검색 실행
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, len(collection_data['ids'])),
                include=["documents", "metadatas", "distances"],
                where={"type": {"$in": ["overview", "market", "challenge"]}}  # 특정 타입만 검색
            )
            
            # 결과 포맷팅
            formatted_results = []
            for i in range(len(results['ids'][0])):
                try:
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None,
                        'relevance_score': 1.0 - (results['distances'][0][i] if 'distances' in results else 0.0)
                    })
                except Exception as e:
                    logger.error(f"결과 포맷팅 중 오류 발생 (인덱스 {i}): {str(e)}")
                    continue
            
            # 관련성 점수로 정렬
            formatted_results.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)
            
            logger.info(f"쿼리 검색 완료: {query} (결과: {len(formatted_results)}개)")
            if formatted_results:
                logger.debug(f"첫 번째 결과: {formatted_results[0]['text'][:100]}...")
                logger.debug(f"관련성 점수: {formatted_results[0]['relevance_score']:.2f}")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"쿼리 검색 중 오류 발생: {str(e)}")
            return []
    
    def _add_default_data(self) -> None:
        """기본 데이터 추가"""
        default_data = [
            {
                'text': 'UAM(Urban Air Mobility)은 도시 내 공중 이동을 위한 새로운 교통 수단으로, 전기 수직이착륙(eVTOL) 항공기를 활용합니다. 주요 기술로는 배터리, 모터, 자율주행 시스템이 있으며, 안전성과 효율성이 핵심 과제입니다.',
                'metadata': {
                    'title': 'UAM 기술 개요',
                    'type': 'overview',
                    'source': '기술 보고서',
                    'key_points': ['eVTOL', '배터리', '자율주행', '안전성']
                }
            },
            {
                'text': 'UAM 시장은 2030년까지 연평균 30% 이상 성장할 것으로 예상되며, 주요 기업들이 시장 진출을 준비 중입니다. 특히 아시아 지역의 성장이 두드러질 것으로 전망됩니다.',
                'metadata': {
                    'title': 'UAM 시장 전망',
                    'type': 'market',
                    'source': '시장 조사 보고서',
                    'key_points': ['시장 성장', '아시아 시장', '기업 진출']
                }
            },
            {
                'text': 'UAM의 주요 기술 과제는 배터리 성능 향상, 소음 감소, 안전성 확보입니다. 특히 배터리 기술은 비행 시간과 안전성에 직접적인 영향을 미치므로 중요합니다.',
                'metadata': {
                    'title': 'UAM 기술 과제',
                    'type': 'challenge',
                    'source': '기술 분석 보고서',
                    'key_points': ['배터리 성능', '소음 감소', '안전성']
                }
            }
        ]
        
        try:
            # 기존 데이터 삭제
            self.clear()
            
            # 새 데이터 추가
            self.add_documents(default_data)
            logger.info("기본 데이터가 추가되었습니다.")
            
            # 추가 확인
            collection_data = self.collection.get()
            logger.info(f"Vector DB 문서 수: {len(collection_data['ids'])}")
            
        except Exception as e:
            logger.error(f"기본 데이터 추가 중 오류 발생: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """여러 문서를 Vector DB에 추가"""
        try:
            # 문서 ID 생성
            ids = [f"doc_{i}" for i in range(len(documents))]
            
            # 메타데이터 추출 및 포맷팅
            metadatas = [self._format_metadata(doc.get('metadata', {})) for doc in documents]
            
            # 문서 텍스트 추출
            texts = [doc.get('text', '') for doc in documents]
            
            # 임베딩 생성
            embeddings = self.model.encode(texts).tolist()
            
            # Vector DB에 추가
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
            
            logger.info(f"{len(documents)}개의 문서가 Vector DB에 추가되었습니다.")
            
        except Exception as e:
            logger.error(f"Vector DB 문서 추가 중 오류 발생: {str(e)}")
            raise
    
    def clear(self) -> None:
        """Vector DB의 모든 데이터를 삭제"""
        try:
            self.collection.delete(where={})
            logger.info("Vector DB가 초기화되었습니다.")
        except Exception as e:
            logger.error(f"Vector DB 초기화 중 오류 발생: {str(e)}")
            raise 