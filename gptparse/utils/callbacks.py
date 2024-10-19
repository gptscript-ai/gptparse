from typing import Optional, Union, Any
from uuid import UUID
from tqdm.auto import tqdm
from langchain_core.callbacks import BaseCallbackHandler


class BatchCallback(BaseCallbackHandler):
    def __init__(self, total: int, ai_model: str):
        super().__init__()
        self.count = 0
        self.total = total
        self.progress_bar = tqdm(
            total=total, desc=f"Reading pages using {ai_model}", unit="page"
        )
        self.parent_run_id: Optional[UUID] = None

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        self.progress_bar.close()

    def on_llm_end(
        self,
        response,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs,
    ) -> None:
        self.count += 1
        self.progress_bar.update(1)
        if self.count == self.total:
            self.progress_bar.close()
