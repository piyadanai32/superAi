import logging

logger = logging.getLogger(__name__)

def load_huggingface_pipeline():
    try:
        from transformers import pipeline
        huggingface_qa_pipeline = pipeline("text-generation", model="flax-community/gpt2-base-thai")
        logger.info("โหลดโมเดล Hugging Face สำเร็จ")
        return huggingface_qa_pipeline
    except Exception as e:
        logger.error(f"ไม่สามารถโหลดโมเดล Hugging Face: {str(e)}")
        return None

def ask_huggingface_model(question, huggingface_qa_pipeline):
    """
    ถามคำถามกับโมเดล Hugging Face
    """
    try:
        if huggingface_qa_pipeline is None:
            logger.warning("ไม่สามารถใช้งาน Hugging Face model ได้")
            return "ขออภัย ระบบไม่สามารถโหลดโมเดล AI ได้ กรุณาติดต่อผู้ดูแลระบบ"
        logger.info(f"กำลังถามโมเดล Hugging Face: '{question}'")
        prompt = f"""โปรดตอบคำถามต่อไปนี้อย่างละเอียด:

คำถาม: {question}

คำตอบ: """
        outputs = huggingface_qa_pipeline(
            prompt,
            max_new_tokens=200,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )
        generated_text = outputs[0]['generated_text']
        try:
            answer = generated_text.split("คำตอบ:", 1)[1].strip()
            words = answer.split()
            answer = " ".join(dict.fromkeys(words))
        except:
            answer = generated_text
        logger.info(f"ได้รับคำตอบจาก Hugging Face: {answer[:100]}...")
        if len(answer) < 10 or any(word in answer.lower() for word in ['error', 'ไม่สามารถ', 'ขออภัย']):
            return "ขออภัย ฉันไม่สามารถตอบคำถามนี้ได้อย่างเหมาะสม กรุณาถามคำถามอื่น หรือถามในรูปแบบอื่น"
        return answer
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดกับ Hugging Face: {str(e)}")
        return "ขออภัย ระบบกำลังมีปัญหาในการประมวลผล กรุณาลองใหม่ภายหลัง"
