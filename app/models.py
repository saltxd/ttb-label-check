from pydantic import BaseModel

from app.checks import FieldCheck


class ApplicationData(BaseModel):
    application_id: str = ""
    brand_name: str
    abv: float | None = None


class LabelResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    application_id: str = ""
    overall: str
    fields: dict[str, FieldCheck]
    ocr_text: str = ""
    elapsed_ms: int = 0
    ai_assist_used: bool = False
