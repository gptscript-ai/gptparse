from typing import List, Optional
from pydantic import BaseModel


class Page(BaseModel):
    content: str
    input_tokens: int
    output_tokens: int
    page: int


class GPTParseOutput(BaseModel):
    file_path: str
    provider: str
    model: Optional[str]
    completion_time: float
    input_tokens: int
    output_tokens: int
    pages: List[Page]
    error: Optional[str] = None

    def __str__(self):
        if self.error:
            return f"Error: {self.error}"
        return (
            f"Processed {len(self.pages)} pages in {self.completion_time:.2f} seconds"
        )
