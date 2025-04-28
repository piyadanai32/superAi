import os
import json
import logging
from datetime import datetime
from flask import Flask, request, abort, jsonify
from dotenv import load_dotenv  
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.messaging import (
    TextMessage, FlexMessage, FlexContainer, ReplyMessageRequest,
    QuickReply, QuickReplyItem, MessageAction
)
from linebot.v3.exceptions import InvalidSignatureError
from google.cloud.dialogflow_v2 import SessionsClient
from google.cloud.dialogflow_v2.types import TextInput, QueryInput
from google.protobuf.json_format import MessageToDict

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
    "พูดอีกทีได้ไหมคะ" 
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
    bot_name = "DPA Chatbot"
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
            send_text_message(event.reply_token, reply_text)
        else:
            try:
                # Step 1: Dialogflow
                user_session_id = f"{SESSION_ID}-{user_id}"
                response = detect_intent_texts(DIALOGFLOW_PROJECT_ID, user_session_id, actual_message, 'th')
                
                # แปลง response เป็น dict สำหรับตรวจสอบ
                response_dict = MessageToDict(response._pb)
                logger.info(f"Dialogflow response: {json.dumps(response_dict, indent=2, ensure_ascii=False)[:1000]}")
                
                # เตรียมข้อมูลที่จะส่งกลับ
                messages_to_reply = []
                quick_replies = []
                has_payload = False
                
                # ตรวจสอบและรวบรวมทุกข้อความจาก Dialogflow
                if 'queryResult' in response_dict:
                    # ตรวจสอบหา messages ที่มี quick replies และ payloads
                    if 'fulfillmentMessages' in response_dict['queryResult']:
                        for message in response_dict['queryResult']['fulfillmentMessages']:
                            # ตรวจสอบข้อความปกติ
                            if 'text' in message and 'text' in message['text'] and message['text']['text']:
                                for text in message['text']['text']:
                                    if text and text not in INVALID_DIALOGFLOW_RESPONSES:
                                        text_message = TextMessage(text=text)
                                        messages_to_reply.append(text_message)
                            
                            # Check Quick Replies
                            if 'quickReplies' in message:
                                quick_reply_items = []
                                if 'quickReplies' in message['quickReplies'] and isinstance(message['quickReplies']['quickReplies'], list):
                                    for qr in message['quickReplies']['quickReplies']:
                                        quick_reply_items.append(
                                            QuickReplyItem(
                                                action=MessageAction(
                                                    label=qr[:20],  # LINE limits labels to 20 characters
                                                    text=qr
                                                )
                                            )
                                        )
                                if quick_reply_items:
                                    quick_replies = QuickReply(items=quick_reply_items)
                                    # ใส่ quick replies ในข้อความสุดท้าย (ถ้ามี)
                                    if messages_to_reply:
                                        messages_to_reply[-1].quick_reply = quick_replies
                                    
                            # Check Custom Payload
                            if 'payload' in message:
                                has_payload = True
                                process_payload(message['payload'], messages_to_reply)
                
                # ถ้าไม่มีข้อความใน fulfillmentMessages ให้ใช้ fulfillmentText (ถ้ามี)
                if not messages_to_reply and 'fulfillmentText' in response_dict['queryResult']:
                    text_response = response_dict['queryResult']['fulfillmentText']
                    if text_response and text_response not in INVALID_DIALOGFLOW_RESPONSES:
                        text_message = TextMessage(text=text_response, quick_reply=quick_replies if quick_replies else None)
                        messages_to_reply.append(text_message)
                
                # หากมีอย่างน้อยหนึ่งข้อความให้ส่งทั้งหมด
                if messages_to_reply:
                    # เพิ่ม quick replies ให้กับข้อความสุดท้าย (ถ้ามี quick replies แต่ยังไม่ได้ใส่)
                    if quick_replies and not messages_to_reply[-1].quick_reply:
                        messages_to_reply[-1].quick_reply = quick_replies
                    
                    send_multiple_messages(event.reply_token, messages_to_reply)
                else:
                    # ไม่มีข้อความจาก Dialogflow ใช้การค้นหาเอกสาร
                    logger.info("No valid response from Dialogflow, trying document search")
                    reply_text = search_from_documents(actual_message)

                    if not reply_text:
                        # ไม่พบในเอกสาร ใช้ Hugging Face
                        logger.info("No response from documents, trying Hugging Face model")
                        reply_text = ask_huggingface_model(actual_message)
                    
                    # ส่งข้อความที่ได้
                    text_message = TextMessage(text=reply_text, quick_reply=quick_replies if quick_replies else None)
                    send_multiple_messages(event.reply_token, [text_message])
                
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                send_text_message(event.reply_token, "ขออภัย เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้ง")

# ฟังก์ชันใหม่เพื่อประมวลผล payload และเพิ่มเข้าไปในรายการข้อความ
def process_payload(payload, messages_list):
    try:
        logger.info(f"Processing payload: {json.dumps(payload, indent=2, ensure_ascii=False)[:500]}")
        
        # ตรวจสอบว่ามี LINE Flex Message หรือไม่
        if 'line' in payload and isinstance(payload['line'], dict):
            line_content = payload['line']
            
            if 'type' in line_content and line_content['type'] == 'flex':
                # สร้าง Flex Message
                flex_message = create_flex_message(line_content)
                if flex_message:
                    messages_list.append(flex_message)
                return
        
        # ตรวจสอบกรณีที่ payload เป็น Flex Message โดยตรง
        if isinstance(payload, dict) and 'type' in payload and payload['type'] == 'flex':
            # สร้าง Flex Message
            flex_message = create_flex_message(payload)
            if flex_message:
                messages_list.append(flex_message)
            return
            
    except Exception as e:
        logger.error(f"Error processing payload: {str(e)}")

# สร้าง Flex Message จาก payload
def create_flex_message(flex_content):
    try:
        # Debug
        logger.info(f"Creating Flex Message: {json.dumps(flex_content)[:200]}...")
        
        # ตรวจสอบและแก้ไขโครงสร้าง JSON ที่อาจมีปัญหา
        if 'contents' in flex_content:
            # แก้ไขปัญหาที่อาจมีในโครงสร้าง JSON
            flex_contents = flex_content['contents']
            # ทำความสะอาดข้อมูล JSON (ถ้าจำเป็น)
            if isinstance(flex_contents, dict):
                # ตรวจสอบความถูกต้องของ JSON
                json.dumps(flex_contents)  # ทดสอบว่าสามารถแปลงเป็น JSON ได้

            # สร้าง FlexContainer object (จำเป็นสำหรับ LINE API v3)
            flex_container = FlexContainer.from_dict(flex_content['contents'])
            
            # สร้างและส่งคืน FlexMessage object
            return FlexMessage(
                alt_text=flex_content.get('altText', 'Flex Message'),
                contents=flex_container
            )
        
        return None
    except Exception as e:
        logger.error(f"Error creating Flex Message: {str(e)}")
        return None

# ฟังก์ชั่นส่งหลายข้อความในครั้งเดียว
def send_multiple_messages(reply_token, messages):
    try:
        if not messages:
            logger.warning("No messages to send")
            return
            
        logger.info(f"Sending {len(messages)} messages")
        reply_request = ReplyMessageRequest(
            reply_token=reply_token,
            messages=messages
        )
        line_bot_api.reply_message_with_http_info(reply_request)
        logger.info("Messages sent successfully")
    except Exception as e:
        logger.error(f"Error sending multiple messages: {str(e)}")
        # Fallback to simple text message
        try:
            send_text_message(reply_token, "ขออภัย เกิดข้อผิดพลาดในการส่งข้อความ")
        except:
            logger.error("Failed to send fallback message")

# ฟังก์ชั่นส่งข้อความตัวอักษร (แยกออกมาเพื่อความสะดวก)
def send_text_message(reply_token, text):
    try:
        # ตรวจสอบว่าข้อความไม่เป็นค่าว่าง
        text = text if text else "ขออภัย ไม่พบข้อมูล"
        
        # ตัดความยาวข้อความถ้าเกิน limit
        if len(text) > 2000:
            text = text[:1997] + "..."
            
        logger.info(f"Sending text reply: {text[:100]}...")
        reply_request = ReplyMessageRequest(
            reply_token=reply_token,
            messages=[TextMessage(text=text)]
        )
        line_bot_api.reply_message_with_http_info(reply_request)
    except Exception as e:
        logger.error(f"Error sending text message: {str(e)}")

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
                fulfillment_messages = []
            query_result = MockQueryResult()
            _pb = None
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
        # ตัดข้อความให้สั้นลงเพื่อหลีกเลี่ยงปัญหาความยาวเกินขีดจำกัด
        if len(result) > 2000:
            result = result[:1997] + "..."
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