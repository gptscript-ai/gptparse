from dataclasses import dataclass
from typing import List
import os


@dataclass
class Page:
    content: str
    input_tokens: int
    output_tokens: int
    page: int


@dataclass
class GPTParseOutput:
    file_path: str
    provider: str
    model: str
    completion_time: float
    input_tokens: int
    output_tokens: int
    pages: List[Page]

    @property
    def file_name(self):
        return os.path.basename(self.file_path)

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens

    @property
    def average_tokens_per_page(self):
        return self.total_tokens / len(self.pages) if self.pages else 0
