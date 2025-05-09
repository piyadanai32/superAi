import json
import logging
from linebot.v3.messaging import (
    TextMessage, FlexMessage, FlexContainer, ReplyMessageRequest,
    QuickReply, QuickReplyItem, MessageAction
)

logger = logging.getLogger(__name__)

def process_payload(payload, messages_list):
    try:
        logger.info(f"กำลังประมวลผล payload: {json.dumps(payload, indent=2, ensure_ascii=False)[:500]}")
        if 'line' in payload and isinstance(payload['line'], dict):
            line_content = payload['line']
            if 'type' in line_content and line_content['type'] == 'flex':
                flex_message = create_flex_message(line_content)
                if flex_message:
                    messages_list.append(flex_message)
                return
        if isinstance(payload, dict) and 'type' in payload and payload['type'] == 'flex':
            flex_message = create_flex_message(payload)
            if flex_message:
                messages_list.append(flex_message)
            return
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการประมวลผล payload: {str(e)}")

def create_flex_message(flex_content):
    try:
        logger.info(f"กำลังสร้าง Flex Message: {json.dumps(flex_content)[:200]}...")
        if 'contents' in flex_content:
            flex_contents = flex_content['contents']
            if isinstance(flex_contents, dict):
                json.dumps(flex_contents)
            flex_container = FlexContainer.from_dict(flex_content['contents'])
            return FlexMessage(
                alt_text=flex_content.get('altText', 'Flex Message'),
                contents=flex_container
            )
        return None
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการสร้าง Flex Message: {str(e)}")
        return None

def send_multiple_messages(line_bot_api, reply_token, messages):
    try:
        if not messages:
            logger.warning("ไม่มีข้อความที่จะส่ง")
            return
        logger.info(f"กำลังส่ง {len(messages)} ข้อความ")
        reply_request = ReplyMessageRequest(
            reply_token=reply_token,
            messages=messages
        )
        line_bot_api.reply_message_with_http_info(reply_request)
        logger.info("ส่งข้อความสำเร็จ")
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการส่งข้อความหลายรายการ: {str(e)}")
        try:
            send_text_message(line_bot_api, reply_token, "ขออภัย เกิดข้อผิดพลาดในการส่งข้อความ")
        except:
            logger.error("ไม่สามารถส่งข้อความสำรองได้")

def send_text_message(line_bot_api, reply_token, text):
    try:
        text = text if text else "ขออภัย ไม่พบข้อมูล"
        if len(text) > 4997:
            text = text[:4997] + "..."
        logger.info(f"กำลังส่งข้อความตอบกลับ: {text[:100]}...")
        reply_request = ReplyMessageRequest(
            reply_token=reply_token,
            messages=[TextMessage(text=text)]
        )
        line_bot_api.reply_message_with_http_info(reply_request)
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการส่งข้อความตัวอักษร: {str(e)}")
