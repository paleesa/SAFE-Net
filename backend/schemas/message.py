from pydantic import BaseModel

class SendMessageRequest(BaseModel):
    sender_id: str
    receiver_id: str
    message_text: str