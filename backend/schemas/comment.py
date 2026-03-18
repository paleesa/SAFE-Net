from pydantic import BaseModel

class CreateCommentRequest(BaseModel):
    user_id: str
    post_id: str
    comment_text: str