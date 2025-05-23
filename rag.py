import json
import logging
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, model_name: str = 'intfloat/multilingual-e5-base'):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.dimension = None
        logger.info(f"Initialized RAG system with model: {model_name}")

    def load_documents(self, json_path: str):
        """Load documents from a JSON file or directory containing JSON files"""
        try:
            json_files = []
            if os.path.isdir(json_path):
                # ถ้าเป็นโฟลเดอร์ ให้หาไฟล์ .json ทั้งหมด
                for file in os.listdir(json_path):
                    if file.endswith('.json'):
                        json_files.append(os.path.join(json_path, file))
            else:
                # ถ้าเป็นไฟล์เดี่ยว
                json_files = [json_path]
            
            processed_docs = []
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Prepare documents for indexing from each file
                    for item in data:
                        if isinstance(item, dict) and "question" in item and "answer" in item:
                            text = f"{item['question']} {item['answer']}"
                            processed_docs.append({
                                'text': text,
                                'question': item['question'],
                                'answer': item['answer'],
                                'source': os.path.basename(json_file)  # เก็บชื่อไฟล์ต้นทาง
                            })
                except Exception as e:
                    logger.error(f"Error loading file {json_file}: {str(e)}")
                    continue
            
            self.documents = processed_docs
            self._build_index()
            logger.info(f"Loaded {len(self.documents)} documents from {len(json_files)} files")
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
        embeddings = self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        
        # Initialize FAISS index
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype('float32'))
        logger.info(f"Built FAISS index with {len(self.documents)} documents")

    def search(self, query: str, k: int = 3) -> List[Dict]:
        try:
            if self.index is None:
                return []

            query_embedding = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            scores, indices = self.index.search(query_embedding.astype('float32'), k)

            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0 and idx < len(self.documents):
                    doc = self.documents[idx]
                    results.append({
                        'question': doc['question'],
                        'answer': doc['answer'],
                        'score': float(score),
                        'text': doc['text']
                    })

            # Sort by score
            results.sort(key=lambda x: x['score'], reverse=True)
            logger.info(f"Query: {query}")
            logger.info(f"Top result: {results[0]['question']} (score: {results[0]['score']:.4f})")
            
            return results

        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []
