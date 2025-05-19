from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass

@dataclass
class SearchResult:
    text: str
    metadata: Dict[str, Any]
    score: float
    source: str

class HybridRetriever:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = []
        
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """문서를 추가하고 임베딩을 생성합니다."""
        texts = [doc["text"] for doc in documents]
        self.embeddings = self.model.encode(texts)
        self.documents = documents
        
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """하이브리드 검색을 수행합니다."""
        # 쿼리 임베딩 생성
        query_embedding = self.model.encode(query)
        
        # 코사인 유사도 계산
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # 상위 k개 결과 선택
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append(SearchResult(
                text=doc["text"],
                metadata=doc.get("metadata", {}),
                score=float(similarities[idx]),
                source=doc.get("source", "unknown")
            ))
            
        return results 