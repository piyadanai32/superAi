import os
import json
import logging
import re
from datetime import datetime
from flask import Flask, request, abort, jsonify
from dotenv import load_dotenv  
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.messaging import (
    TextMessage, FlexMessage, QuickReply, QuickReplyItem, MessageAction
)
from linebot.v3.exceptions import InvalidSignatureError
from google.protobuf.json_format import MessageToDict

from retriever import search_from_documents
from dialogflow import detect_intent_texts
from message import (
    process_payload, create_flex_message,
    send_multiple_messages, send_text_message
)

# โหลด environment variables จากไฟล์ .env
load_dotenv()

# ตรวจสอบและกำหนดค่าตัวแปรสภาพแวดล้อมที่จำเป็น
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not GOOGLE_APPLICATION_CREDENTIALS:
    raise ValueError("ไม่พบตัวแปรสภาพแวดล้อม GOOGLE_APPLICATION_CREDENTIALS")

DIALOGFLOW_PROJECT_ID = os.getenv("DIALOGFLOW_PROJECT_ID")
if not DIALOGFLOW_PROJECT_ID:
    raise ValueError("ไม่พบตัวแปรสภาพแวดล้อม DIALOGFLOW_PROJECT_ID")

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
if not LINE_CHANNEL_ACCESS_TOKEN:
    raise ValueError("ไม่พบตัวแปรสภาพแวดล้อม LINE_CHANNEL_ACCESS_TOKEN")

LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
if not LINE_CHANNEL_SECRET:
    raise ValueError("ไม่พบตัวแปรสภาพแวดล้อม LINE_CHANNEL_SECRET")

# กำหนดค่า Config
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
SESSION_ID = "line-bot-session"

# Flask App
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = app.logger

# Line Bot
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)

# คำตอบที่ไม่ต้องการจาก Dialogflow (หากได้คำตอบเหล่านี้จะถือว่า Dialogflow ไม่สามารถตอบคำถามได้)
INVALID_DIALOGFLOW_RESPONSES = [
    "ขอโทษค่ะ พูดอีกครั้งได้ไหมคะ",
    "ขอโทษค่ะ ไม่เข้าใจ",
    "พูดใหม่อีกครั้งได้ไหมคะ",
    "ขอโทษค่ะ ฉันไม่เข้าใจค่ะ",
    "พูดอีกทีได้ไหมคะ" 
]

@app.route("/callback", methods=['POST'])
def callback():
    """
    เส้นทาง webhook สำหรับรับข้อความจาก LINE
    """
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    logger.info(f"ได้รับคำขอ: {body}")

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logger.error("ลายเซ็นไม่ถูกต้อง")
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    """
    จัดการข้อความที่ได้รับจากผู้ใช้ LINE
    """
    user_id = event.source.user_id
    text_from_user = event.message.text
    logger.info(f"ข้อความจาก {user_id}: {text_from_user}")

    # ตรวจสอบว่าเป็นข้อความในกลุ่มหรือห้องสนทนาหรือไม่
    is_group = hasattr(event.source, 'type') and event.source.type in ['group', 'room']
    bot_name = "DPA Chatbot"
    should_respond = False
    actual_message = text_from_user

    # ถ้าเป็นข้อความในกลุ่ม ตรวจสอบว่าเรียกบอทหรือไม่
    if is_group and text_from_user.startswith(f'@{bot_name}'):
        actual_message = text_from_user.split(f'@{bot_name}', 1)[-1].strip()
        should_respond = True
        logger.info(f"ตรวจพบข้อความในกลุ่ม แยกข้อความได้: {actual_message}")
    elif not is_group:
        # ถ้าเป็นการแชทส่วนตัว ตอบทุกข้อความ
        should_respond = True

    if should_respond:
        if not actual_message:
            # กรณีที่ข้อความว่างเปล่า (เช่น เพียงแค่เรียกชื่อบอทในกลุ่ม)
            reply_text = f"สวัสดีค่ะ หนูชื่อ {bot_name} คุณต้องการสอบถามอะไรค่ะ?"
            send_text_message(line_bot_api, event.reply_token, reply_text)
        else:
            try:
                # ขั้นตอนที่ 1: ส่งคำถามไปยัง Dialogflow
                logger.info("เริ่มขั้นตอนที่ 1: ส่งคำถามไปยัง Dialogflow")
                user_session_id = f"{SESSION_ID}-{user_id}"
                response = detect_intent_texts(DIALOGFLOW_PROJECT_ID, user_session_id, actual_message, 'th')
                
                # แปลง response เป็น dict สำหรับตรวจสอบ
                response_dict = MessageToDict(response._pb)
                logger.info(f"คำตอบจาก Dialogflow: {json.dumps(response_dict, indent=2, ensure_ascii=False)[:1000]}")
                
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
                            
                            # ตรวจสอบ Quick Replies
                            if 'quickReplies' in message:
                                quick_reply_items = []
                                if 'quickReplies' in message['quickReplies'] and isinstance(message['quickReplies']['quickReplies'], list):
                                    for qr in message['quickReplies']['quickReplies']:
                                        quick_reply_items.append(
                                            QuickReplyItem(
                                                action=MessageAction(
                                                    label=qr[:20],  # LINE จำกัดความยาวของป้ายชื่อไม่เกิน 20 ตัวอักษร
                                                    text=qr
                                                )
                                            )
                                        )
                                if quick_reply_items:
                                    quick_replies = QuickReply(items=quick_reply_items)
                                    # ใส่ quick replies ในข้อความสุดท้าย (ถ้ามี)
                                    if messages_to_reply:
                                        messages_to_reply[-1].quick_reply = quick_replies
                                    
                            # ตรวจสอบ Custom Payload (เช่น Flex Message)
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
                    logger.info("พบคำตอบจาก Dialogflow ส่งคำตอบให้ผู้ใช้")
                    # เพิ่ม quick replies ให้กับข้อความสุดท้าย (ถ้ามี quick replies แต่ยังไม่ได้ใส่)
                    if quick_replies and not messages_to_reply[-1].quick_reply:
                        messages_to_reply[-1].quick_reply = quick_replies
                    
                    send_multiple_messages(line_bot_api, event.reply_token, messages_to_reply)
                else:
                    # ขั้นตอนที่ 2: ค้นหาในเอกสาร
                    logger.info("เริ่มขั้นตอนที่ 2: ค้นหาในเอกสาร")
                    reply_text, found_in_docs, rag_context = search_from_documents(actual_message)

                    # ส่งข้อความที่ได้
                    text_message = TextMessage(text=reply_text, quick_reply=quick_replies if quick_replies else None)
                    send_multiple_messages(line_bot_api, event.reply_token, [text_message])
                
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในการประมวลผลข้อความ: {str(e)}")
                send_text_message(line_bot_api, event.reply_token, "ขออภัย เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้ง")

@app.route("/")
def home():
    """
    หน้าแรกเพื่อตรวจสอบสถานะการทำงานของบอท
    """
    return "LINE Bot with Dialogflow + Document + Hugging Face กำลังทำงาน!"

@app.route("/status")
def status():
    """
    แสดงสถานะของบอทและส่วนประกอบต่าง ๆ ในรูปแบบ JSON
    """
    doc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents.json")
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    status_report = {
        "dialogflow": {
            "status": "✅ พร้อมใช้งาน" if DIALOGFLOW_PROJECT_ID and os.path.exists(cred_path) else "❌ ยังไม่ได้ตั้งค่า",
            "project_id": DIALOGFLOW_PROJECT_ID,
            "credentials": "พบไฟล์" if os.path.exists(cred_path) else "ไม่พบไฟล์"
        },
        "documents": {
            "status": "✅ พร้อมใช้งาน" if os.path.exists(doc_path) else "❌ ไม่พบไฟล์",
            "path": doc_path,
            "exists": os.path.exists(doc_path)
        },
        "line_api": {
            "status": "✅ ตั้งค่าแล้ว" if LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET else "❌ ยังไม่ได้ตั้งค่า"
        }
    }

    return jsonify({
        "status": "online",
        "components": status_report,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)