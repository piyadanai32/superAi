import logging
from google.cloud.dialogflow_v2 import SessionsClient
from google.cloud.dialogflow_v2.types import TextInput, QueryInput

logger = logging.getLogger(__name__)

def detect_intent_texts(project_id, session_id, text, language_code):
    """
    ส่งข้อความไปยัง Dialogflow เพื่อตรวจจับเจตนา (intent)
    """
    try:
        logger.info(f"กำลังติดต่อ Dialogflow: Project={project_id}, Session={session_id}")
        session_client = SessionsClient()
        session = session_client.session_path(project_id, session_id)
        text_input = TextInput(text=text, language_code=language_code)
        query_input = QueryInput(text=text_input)
        response = session_client.detect_intent(request={"session": session, "query_input": query_input})
        return response
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดกับ Dialogflow: {str(e)}")
        # สร้าง response จำลองเพื่อให้โค้ดยังทำงานต่อได้
        class MockResponse:
            class MockQueryResult:
                fulfillment_text = ""
                fulfillment_messages = []
            query_result = MockQueryResult()
            _pb = type('MockPb', (object,), {})()
        return MockResponse()
