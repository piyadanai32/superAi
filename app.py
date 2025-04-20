import os
import json
import logging
from datetime import datetime
from flask import Flask, request, abort, jsonify
from dotenv import load_dotenv  
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.messaging import TextMessage, ReplyMessageRequest
from linebot.v3.exceptions import InvalidSignatureError
from google.cloud.dialogflow_v2 import SessionsClient
from google.cloud.dialogflow_v2.types import TextInput, QueryInput

# โหลด environment variables
load_dotenv()

# กำหนดค่า Config
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
DIALOGFLOW_PROJECT_ID = os.getenv("DIALOGFLOW_PROJECT_ID")
SESSION_ID = "line-bot-session"

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

# Flask App
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = app.logger

# Line Bot
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)

# โหลด Hugging Face model
huggingface_qa_pipeline = None
try:
    from transformers import pipeline
    # เปลี่ยนเป็น model ที่รองรับ text-generation
    # huggingface_qa_pipeline = pipeline("text-generation", model="gpt2")
    # หรือใช้ model ภาษาไทย ถ้าต้องการ
    huggingface_qa_pipeline = pipeline("text-generation", model="flax-community/gpt2-base-thai")
    logger.info("Hugging Face model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Hugging Face model: {str(e)}")

# คำตอบที่ไม่ต้องการจาก Dialogflow
INVALID_DIALOGFLOW_RESPONSES = [
    "ขอโทษค่ะ พูดอีกครั้งได้ไหมคะ",
    "ขอโทษค่ะ ไม่เข้าใจ",
    "พูดใหม่อีกครั้งได้ไหมคะ",
    "ขอโทษค่ะ ฉันไม่เข้าใจค่ะ",
    "พูดอีกทีได้ไหมคะ"  # เพิ่มเข้ามาตามที่เห็นในข้อความ error
]

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    logger.info(f"Request body: {body}")

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logger.error("Invalid signature error")
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_id = event.source.user_id
    text_from_user = event.message.text
    logger.info(f"Message from {user_id}: {text_from_user}")

    is_group = hasattr(event.source, 'type') and event.source.type in ['group', 'room']
    bot_name = "น้องสวพ."
    should_respond = False
    actual_message = text_from_user

    if is_group and text_from_user.startswith(f'@{bot_name}'):
        actual_message = text_from_user.split(f'@{bot_name}', 1)[-1].strip()
        should_respond = True
        logger.info(f"Group message detected, extracted message: {actual_message}")
    elif not is_group:
        should_respond = True

    if should_respond:
        if not actual_message:
            reply_text = f"สวัสดีค่ะ หนูชื่อ {bot_name} คุณต้องการสอบถามอะไรค่ะ?"
        else:
            # Step 1: Dialogflow
            user_session_id = f"{SESSION_ID}-{user_id}"
            response = detect_intent_texts(DIALOGFLOW_PROJECT_ID, user_session_id, actual_message, 'th')
            fulfillment_text = response.query_result.fulfillment_text.strip()
            logger.info(f"Dialogflow response: {fulfillment_text}")

            if fulfillment_text and fulfillment_text not in INVALID_DIALOGFLOW_RESPONSES:
                reply_text = fulfillment_text
            else:
                # Step 2: Document search
                logger.info("No valid response from Dialogflow, trying document search")
                reply_text = search_from_documents(actual_message)

                if not reply_text:
                    # Step 3: Hugging Face fallback
                    logger.info("No response from documents, trying Hugging Face model")
                    reply_text = ask_huggingface_model(actual_message)

        # ส่งข้อความกลับไป (แก้ไขการใช้ reply_message)
        try:
            logger.info(f"Sending reply: {reply_text}")
            reply_request = ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)]
            )
            line_bot_api.reply_message_with_http_info(reply_request)
        except Exception as e:
            logger.error(f"Error sending reply: {str(e)}")

# -------------------- Dialogflow --------------------
def detect_intent_texts(project_id, session_id, text, language_code):
    try:
        logger.info(f"Contacting Dialogflow: Project={project_id}, Session={session_id}")
        session_client = SessionsClient()
        session = session_client.session_path(project_id, session_id)
        text_input = TextInput(text=text, language_code=language_code)
        query_input = QueryInput(text=text_input)
        response = session_client.detect_intent(request={"session": session, "query_input": query_input})
        return response
    except Exception as e:
        logger.error(f"Dialogflow Error: {str(e)}")
        class MockResponse:
            class MockQueryResult:
                fulfillment_text = ""
            query_result = MockQueryResult()
        return MockResponse()

# -------------------- Document Search --------------------
def search_from_documents(question):
    try:
        doc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents.json")
        
        if not os.path.exists(doc_path):
            logger.error(f"Document file not found at: {doc_path}")
            return "ขออภัย ไม่พบไฟล์เอกสาร"

        with open(doc_path, "r", encoding="utf-8") as f:
            docs = json.load(f)

        logger.info(f"Loaded {len(docs)} documents from documents.json")
        for doc in docs:
            if question.lower() in doc["question"].lower():
                logger.info(f"Found match: {doc['question']}")
                return doc["answer"]

        return ""
    except Exception as e:
        logger.error(f"Document search error: {str(e)}")
        return ""

# -------------------- Hugging Face --------------------
def ask_huggingface_model(question):
    try:
        if huggingface_qa_pipeline is None:
            return "ขออภัย ระบบไม่สามารถโหลดโมเดล AI ได้ กรุณาติดต่อผู้ดูแลระบบ"
        
        logger.info(f"Asking Hugging Face model: {question}")
        result = huggingface_qa_pipeline(question, max_new_tokens=100)[0]['generated_text']
        return result
    except Exception as e:
        logger.error(f"Hugging Face error: {str(e)}")
        return "ขออภัย ระบบกำลังมีปัญหาในการประมวลผล กรุณาลองใหม่ภายหลัง"

# -------------------- Status Pages --------------------
@app.route("/")
def home():
    return "LINE Bot with Dialogflow + Document + Hugging Face is running!"

@app.route("/status")
def status():
    doc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents.json")
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    status_report = {
        "dialogflow": {
            "status": "✅ Ready" if DIALOGFLOW_PROJECT_ID and os.path.exists(cred_path) else "❌ Not configured",
            "project_id": DIALOGFLOW_PROJECT_ID,
            "credentials": "Found" if os.path.exists(cred_path) else "Not found"
        },
        "documents": {
            "status": "✅ Ready" if os.path.exists(doc_path) else "❌ File not found",
            "path": doc_path,
            "exists": os.path.exists(doc_path)
        },
        "huggingface_model": {
            "status": "✅ Ready" if huggingface_qa_pipeline is not None else "❌ Not loaded"
        },
        "line_api": {
            "status": "✅ Configured" if LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET else "❌ Not configured"
        }
    }

    return jsonify({
        "status": "online",
        "components": status_report,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)