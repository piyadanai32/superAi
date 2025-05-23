import os
import json
import logging
from rag import RAGSystem

logger = logging.getLogger(__name__)
rag_system = None

def initialize_rag():
    global rag_system
    try:
        rag_system = RAGSystem()
        base_dir = os.path.abspath(os.path.dirname(__file__))
        json_path = os.path.join(base_dir, 'data', 'json', 'documents.json')
        if os.path.exists(json_path):
            success = rag_system.load_documents(json_path)
            if success:
                logger.info("RAG system initialized successfully")
                return True
        logger.error(f"Documents file not found at {json_path}")
        return False
    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        return False

def search_from_documents(question):
    try:
        global rag_system
        if rag_system is None:
            if not initialize_rag():
                return "ขออภัย ระบบยังไม่พร้อมใช้งาน", False, None

        results = rag_system.search(question, k=3)
        logger.info(f"คำถาม: {question}")
        
        if not results:
            return "ไม่พบข้อมูลที่เกี่ยวข้อง", False, None

        best_match = results[0]
        logger.info(f"คำตอบที่ดีที่สุด: {best_match['question']} (คะแนน: {best_match['score']:.4f})")
        
        # ปรับเกณฑ์การตัดสินใจ
        if best_match['score'] >= 0.5:  # ต้องมีความเหมือนอย่างน้อย 50%
            contexts = []
            for result in results:
                if result['score'] >= 0.3:  # เก็บ context ที่มีความเหมือนอย่างน้อย 30%
                    contexts.append(f"Q: {result['question']}\nA: {result['answer']}")
            
            if contexts:
                confidence = f" (ความมั่นใจ: {best_match['score']:.0%})"
                return best_match['answer'] + confidence, True, {
                    'question': question,
                    'contexts': contexts,
                    'combined_context': "\n\n".join(contexts),
                    'top_score': best_match['score']
                }

        return "ไม่พบข้อมูลที่ตรงกับคำถามเพียงพอ", False, None

    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการค้นหา: {str(e)}")
        return "เกิดข้อผิดพลาดในการค้นหา", False, None
