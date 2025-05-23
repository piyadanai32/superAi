import os
import json
import logging
from rag import RAGSystem
from ollama_client import generate_response

logger = logging.getLogger(__name__)
rag_system = None

def initialize_rag():
    global rag_system
    try:
        rag_system = RAGSystem()
        base_dir = os.path.abspath(os.path.dirname(__file__))
        json_dir = os.path.join(base_dir, 'data', 'json')
        
        if os.path.exists(json_dir):
            success = rag_system.load_documents(json_dir)  # ส่งโฟลเดอร์แทนไฟล์เดียว
            if success:
                logger.info("RAG system initialized successfully")
                return True
        logger.error(f"JSON directory not found at {json_dir}")
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
            return "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้อง", False, None

        best_match = results[0]
        logger.info(f"คำตอบที่ดีที่สุด: {best_match['question']} (คะแนน: {best_match['score']:.4f})")
        
        if best_match['score'] >= 0.3:
            contexts = []
            for result in results:
                if result['score'] >= 0.2:
                    contexts.append(f"Q: {result['question']}\nA: {result['answer']}")
            
            if contexts:
                combined_context = "\n\n".join(contexts)
                try:
                    answer = generate_response(question, combined_context)
                except:
                    # ถ้า Ollama ไม่พร้อม ใช้คำตอบจาก RAG โดยตรง
                    answer = best_match['answer']
                return answer, True, {
                    'question': question,
                    'contexts': contexts,
                    'combined_context': combined_context,
                    'top_score': best_match['score']
                }

        return "ขออภัย ไม่พบข้อมูลที่ตรงกับคำถามของคุณ", False, None

    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการค้นหา: {str(e)}")
        return "เกิดข้อผิดพลาดในการค้นหา", False, None
