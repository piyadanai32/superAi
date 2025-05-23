import logging
import torch

logger = logging.getLogger(__name__)

def load_huggingface_pipeline():
    try:
        from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
        
        model_name = "flax-community/gpt2-base-thai"
        logger.info(f"กำลังโหลดโมเดล {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        huggingface_qa_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=200,
            device=-1,  # CPU
            pad_token_id=tokenizer.eos_token_id
        )
        
        logger.info(f"โหลดโมเดล {model_name} สำเร็จ")
        return huggingface_qa_pipeline
    except Exception as e:
        logger.error(f"ไม่สามารถโหลดโมเดล: {str(e)}")
        return None

def clean_generated_text(text: str) -> str:
    # ลบอักขระพิเศษและการซ้ำซ้อน
    text = text.replace("<_>", "").replace("_", "")
    words = text.split()
    # ลบคำที่ซ้ำกันติดกัน
    clean_words = []
    prev_word = None
    for word in words:
        if word != prev_word:
            clean_words.append(word)
            prev_word = word
    return " ".join(clean_words)

def ask_huggingface_model(question, huggingface_qa_pipeline, rag_result=None):
    """
    ถามคำถามกับโมเดล Hugging Face
    """
    try:
        if huggingface_qa_pipeline is None:
            logger.warning("ไม่สามารถใช้งาน Hugging Face model ได้")
            return "ขออภัย ระบบไม่สามารถโหลดโมเดล AI ได้ กรุณาติดต่อผู้ดูแลระบบ"
            
        logger.info(f"กำลังถามโมเดล Hugging Face: '{question}'")
        
        if rag_result and 'contexts' in rag_result:
            # ใช้คำตอบที่ได้จาก RAG โดยตรง
            return rag_result['contexts'][0].split('A: ')[1]
        
        prompt = f"คำถาม: {question}\nคำตอบ: "
        
        outputs = huggingface_qa_pipeline(
            prompt,
            max_new_tokens=100,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

        generated_text = outputs[0]['generated_text']
        
        try:
            answer = generated_text.split("คำตอบ: ")[1].strip()
            answer = clean_generated_text(answer)
        except:
            answer = clean_generated_text(generated_text)

        if len(answer) < 10:
            return "ขออภัย ฉันไม่สามารถตอบคำถามนี้ได้อย่างเหมาะสม"
            
        return answer
        
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดกับ Hugging Face: {str(e)}")
        return "ขออภัย ระบบกำลังมีปัญหาในการประมวลผล"
