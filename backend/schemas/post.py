from pydantic import BaseModel
from typing import Optional

class CreatePostRequest(BaseModel):
    user_id: str
    caption_text: str
    media_url: Optional[str] = None