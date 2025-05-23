import json
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, model_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.dimension = None
        logger.info(f"Initialized RAG system with model: {model_name}")

    def load_documents(self, json_path: str):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Prepare documents for indexing
            processed_docs = []
            for item in data:
                if isinstance(item, dict) and "question" in item and "answer" in item:
                    # Combine question and answer for better context
                    text = f"{item['question']} {item['answer']}"
                    processed_docs.append({
                        'text': text,
                        'question': item['question'],
                        'answer': item['answer']
                    })
            
            self.documents = processed_docs
            self._build_index()
            logger.info(f"Loaded {len(self.documents)} documents from {json_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            return False

    def _build_index(self):
        if not self.documents:
            logger.warning("No documents to index")
            return

        # Encode all documents
        texts = [doc['text'] for doc in self.documents]
        embeddings = self.encoder.encode(texts, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy()

        # Initialize FAISS index
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        logger.info(f"Built FAISS index with {len(self.documents)} documents")

    def preprocess_text(self, text: str) -> str:
        """ทำความสะอาดและเตรียมข้อความสำหรับการค้นหา"""
        # ลบช่องว่างที่ไม่จำเป็น
        text = ' '.join(text.split())
        # แปลงเป็นตัวพิมพ์เล็ก
        return text.lower().strip()
    
    def calculate_similarity(self, query_vector, doc_vector) -> float:
        """คำนวณความเหมือนระหว่างข้อความ"""
        from numpy import dot
        from numpy.linalg import norm
        
        # ใช้ cosine similarity
        cos_sim = dot(query_vector, doc_vector) / (norm(query_vector) * norm(doc_vector))
        return float(cos_sim)

    def search(self, query: str, k: int = 3) -> List[Dict]:
        try:
            if self.index is None:
                return []

            # เตรียมคำถาม
            query = self.preprocess_text(query)
            
            # Encode query
            query_vector = self.encoder.encode([query], convert_to_tensor=True)
            query_vector = query_vector.cpu().numpy().astype('float32')

            # Search in FAISS index
            distances, indices = self.index.search(query_vector, k)
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx >= 0 and idx < len(self.documents):
                    doc = self.documents[idx]
                    # คำนวณ similarity score
                    similarity = self.calculate_similarity(
                        query_vector[0],
                        self.encoder.encode(doc['text'], convert_to_tensor=True).cpu().numpy()
                    )
                    
                    results.append({
                        'question': doc['question'],
                        'answer': doc['answer'],
                        'score': similarity,
                        'text': doc['text']
                    })
            
            # เรียงลำดับตาม similarity score
            results.sort(key=lambda x: x['score'], reverse=True)
            logger.info(f"Query: {query}")
            logger.info(f"Top result: {results[0]['question']} (score: {results[0]['score']:.4f})")
            
            return results

        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []
