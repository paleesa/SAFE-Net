from pydantic import BaseModel
from typing import Optional

class SubmitEKYCRequest(BaseModel):
    user_id: str
    front_image_url: Optional[str] = None
    back_image_url: Optional[str] = None
    extracted_text: str