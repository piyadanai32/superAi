import logging
import requests

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

def generate_response(question: str, context: str = None) -> str:
    try:
        if context:
            prompt = f"""คุณเป็น AI ที่ช่วยตอบคำถามโดยใช้ข้อมูลที่ให้มา

ข้อมูลอ้างอิง:
{context}

คำถาม: {question}

คำตอบ:"""
        else:
            prompt = f"""คุณเป็น AI ผู้ช่วยตอบคำถาม

คำถาม: {question}

คำตอบ:"""

        try:
            response = requests.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 512
                }
            }, timeout=30)

            if response.status_code == 200:
                return response.json()["response"].strip()
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return "ขออภัย ระบบยังไม่พร้อมใช้งาน กรุณาลองใหม่ภายหลัง"

        except requests.exceptions.ConnectionError:
            logger.error("ไม่สามารถเชื่อมต่อกับ Ollama server ได้")
            return "ขออภัย ระบบ AI ยังไม่พร้อมใช้งาน กรุณารอสักครู่"
            
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "ขออภัย เกิดข้อผิดพลาดในการประมวลผล"
