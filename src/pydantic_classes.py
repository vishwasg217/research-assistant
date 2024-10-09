from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class Document(BaseModel):
    uuid: int
    title: str
    abstract: str
    categories: list[str]
    authors: list[str]
    create_date: datetime
    update_date: datetime
    paper_url: str
    html_url: str
    paper_id: str
    doi: Optional[str] = None
    report_no: Optional[str] = None
    journal_ref: Optional[str] = None
    license: Optional[str] = None
    comments: Optional[str] = None
    metadata: Optional[dict] = None
