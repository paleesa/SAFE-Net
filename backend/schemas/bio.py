from pydantic import BaseModel

class UpdateBioRequest(BaseModel):
    user_id: str
    bio_text: str