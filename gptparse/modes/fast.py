import os
import time
import logging
from typing import Optional
import pymupdf4llm
from ..outputs import GPTParseOutput, Page
from ..config import setup_logging
import re

setup_logging()


def clean_markdown_content(content: str) -> str:
    """Clean up markdown content from common artifacts."""
    # Remove stray underscores around text
    content = re.sub(r"_([^_]+)_\n", r"\1\n", content)

    # Remove duplicate newlines
    content = re.sub(r"\n{3,}", "\n\n", content)

    # Remove horizontal rules that are just dashes
    content = re.sub(r"\n-{3,}\n", "\n\n", content)

    return content.strip()


def fast(
    file_path: str,
    output_file: Optional[str] = None,
    select_pages: Optional[str] = None,
) -> GPTParseOutput:
    """Convert PDF to Markdown using pymupdf4llm."""
    try:
        start_time = time.time()

        # Parse page selection if provided
        pages = None
        if select_pages:
            pages = []
            for part in select_pages.split(","):
                if "-" in part:
                    start, end = map(int, part.split("-"))
                    pages.extend(range(start - 1, end))
                else:
                    pages.append(int(part) - 1)

        # Convert PDF to markdown
        markdown_text = pymupdf4llm.to_markdown(
            file_path, pages=pages, page_chunks=True
        )

        # Process results
        processed_pages = []
        for idx, page_data in enumerate(markdown_text):
            page_number = pages[idx] + 1 if pages else idx + 1
            content = clean_markdown_content(page_data["text"])

            # Create page object
            processed_pages.append(
                Page(
                    content=content,
                    input_tokens=0,  # Not applicable for fast mode
                    output_tokens=0,  # Not applicable for fast mode
                    page=page_number,
                )
            )

        completion_time = time.time() - start_time

        result = GPTParseOutput(
            file_path=os.path.abspath(file_path),
            provider="local",
            model="pymupdf4llm",
            completion_time=completion_time,
            input_tokens=0,  # Not applicable for fast mode
            output_tokens=0,  # Not applicable for fast mode
            pages=processed_pages,
        )

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                multiple_pages = len(result.pages) > 1
                for page in result.pages:
                    if multiple_pages:
                        f.write(f"---Page {page.page} Start---\n\n")
                    f.write(f"{page.content}\n\n")
                    if multiple_pages:
                        f.write(f"---Page {page.page} End---\n\n")

        return result

    except Exception as e:
        logging.error(f"Error in fast mode: {str(e)}")
        return GPTParseOutput(
            file_path=os.path.abspath(file_path),
            provider="local",
            model="pymupdf4llm",
            completion_time=0,
            input_tokens=0,
            output_tokens=0,
            pages=[],
            error=str(e),
        )
