from pydantic import BaseModel
from typing import Optional, Literal

class AnalyzeTextRequest(BaseModel):
    user_id: str
    source_type: Literal["bio", "post", "comment"]
    source_id: Optional[str] = None
    text: str