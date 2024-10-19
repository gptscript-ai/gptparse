# -*- coding: utf-8 -*-
import os
import io
import time
import base64
import json
import re
import logging
import warnings
from typing import Optional, List
from tqdm import tqdm
from PIL import Image
from pdf2image import convert_from_bytes
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from ..config import get_config, setup_logging
from ..outputs import GPTParseOutput, Page
from ..models import model_interface
from ..utils.callbacks import BatchCallback
from ..utils.image_utils import resize_image
from ..utils.pdf_utils import split_pdf_into_chunks
from ..models.model_interface import PROVIDER_MODELS

setup_logging()


def parse_page_selection(select_pages: str) -> List[int]:
    if not select_pages:
        return []
    pages = []
    for part in select_pages.split(","):
        if "-" in part:
            start, end = map(int, part.split("-"))
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))
    return [p - 1 for p in pages]


def vision(
    concurrency: int,
    file_path: str,
    model: Optional[str] = None,
    output_file: Optional[str] = None,
    custom_system_prompt: Optional[str] = None,
    select_pages: Optional[str] = None,
    provider: str = "openai",
) -> GPTParseOutput:
    try:
        start_time = time.time()
        config = get_config()
        provider = provider or config.get("provider", "openai")
        model = model or config.get("model") or PROVIDER_MODELS[provider]["default"]

        ai_model = model_interface.get_model(provider, model)
        warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

        pages_to_process = parse_page_selection(select_pages) if select_pages else None

        try:
            pdf_chunks = split_pdf_into_chunks(file_path)
            images = []
            with tqdm(
                total=len(pdf_chunks),
                desc="Checking input document pages",
                unit="chunk",
            ) as pbar:
                for chunk in pdf_chunks:
                    chunk_images = convert_from_bytes(
                        chunk, thread_count=min(4, concurrency), dpi=300
                    )
                    images.extend(chunk_images)
                    pbar.update(1)
        except Exception as e:
            logging.error(f"Error converting PDF to images: {str(e)}")
            return GPTParseOutput(
                file_path=os.path.abspath(file_path),
                provider=provider,
                model=model,
                completion_time=0,
                input_tokens=0,
                output_tokens=0,
                pages=[],
                error=str(e),
            )

        total_pages = len(images)

        if not pages_to_process:
            pages_to_process = range(total_pages)
        else:
            pages_to_process = [p for p in pages_to_process if p < total_pages]

        batch_messages = []
        for i in tqdm(
            pages_to_process, desc="Preparing pages for vision input", unit="page"
        ):
            # Resize the image
            resized_image = resize_image(images[i])

            # Convert resized image to base64
            buffered = io.BytesIO()
            resized_image.save(buffered, format="PNG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Prepare the message for the AI model
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Convert the content of the image into markdown format, ensuring the appropriate structure for various components including tables, lists, and other images. You will not add any of your own commentary to your response. Consider the following:\n\n- **Tables:** If the image contains tables, convert them into markdown tables. Ensure that all columns and rows from the table are accurately captured. Do not convert tables into JSON unless every column and row, with all data, can be properly represented.\n- **Lists:** If the image contains lists, convert them into markdown lists.\n- **Images:** If the image contains other images, summarize each image into text and wrap it with `<image></image>` tags.\n\n# Steps\n\n1. **Image Analysis:** Identify the various elements in the image such as tables, lists, and other images.\n   \n2. **Markdown Conversion:**\n   - For tables, use the markdown format for tables. Make sure all columns and rows are preserved, including headers and any blank cells.\n   - For lists, use markdown list conventions (ordered or unordered as per the original).\n   - For images, write a brief descriptive summary of the image content and wrap it using `<image></image>` tags.\n\n3. **Compile:** Assemble all converted elements into cohesive markdown-formatted text.\n\n# Output Format\n\n- The output should be in markdown format, accurately representing each element from the image with appropriate markdown syntax. Pay close attention to the structure of tables, ensuring that no columns or rows are omitted.\n\n# Examples\n\n**Input Example 1:**\n\nAn image containing a table with five columns and three rows, a list, and another image.\n\n**Output Example 1:**\n\n```\n| Column 1 | Column 2 | Column 3 | Column 4 | Column 5 |\n| -------- | -------- | -------- | -------- | -------- |\n| Row 1    | Data 2   | Data 3   | Data 4   | Data 5   |\n| Row 2    | Data 2   | Data 3   | Data 4   |          |\n| Row 3    | Data 2   |          | Data 4   | Data 5   |\n\n- List Item 1\n- List Item 2\n- List Item 3\n\n<image></image>\nImage description with as much detail as possible here.\n</image>\n```\n\n# Notes\n\n- Ensure that the markdown syntax is correct and renders well when processed.\n- Preserve column and row structure for tables, ensuring no data is lost or misrepresented.\n- Be attentive to the layout and order of elements as they appear in the image.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                    },
                ]
            )
            batch_messages.append([message])

        cb = BatchCallback(len(batch_messages), f"{provider}/{model}")
        try:
            # Process batch messages
            results = ai_model.batch(
                batch_messages,
                config=RunnableConfig(max_concurrency=concurrency, callbacks=[cb]),
            )
        except Exception as e:
            error_msg = str(e)
            if any(
                keyword in error_msg.lower()
                for keyword in [
                    "authentication_error",
                    "invalid_api_key",
                    "api key not valid",
                ]
            ):
                if provider == "openai":
                    error_dict = json.loads(error_msg.split(" - ", 1)[1])
                    error_message = error_dict["error"]["message"]
                elif provider == "google":
                    error_message = re.search(
                        r"Invalid argument provided to Gemini: (.+)", error_msg
                    ).group(1)
                else:  # For other providers like Anthropic
                    error_dict = json.loads(error_msg.split(" - ", 1)[1])
                    error_message = error_dict["error"]["message"]
                raise ValueError(
                    f"Authentication error for {provider}: {error_message}"
                )
            raise e

        # Process results
        processed_pages = []
        total_input_tokens = 0
        total_output_tokens = 0

        for i, result in enumerate(results):
            markdown_content = result.content
            input_tokens = result.usage_metadata.get("input_tokens", 0)
            output_tokens = result.usage_metadata.get("output_tokens", 0)

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            processed_pages.append(
                Page(
                    content=markdown_content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    page=pages_to_process[i] + 1,
                )
            )

        completion_time = time.time() - start_time

        result = GPTParseOutput(
            file_path=os.path.abspath(file_path),
            provider=provider,
            model=model,
            completion_time=completion_time,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            pages=processed_pages,
        )

        if output_file:
            if not output_file.lower().endswith((".md", ".txt")):
                raise ValueError("Output file must have a .md or .txt extension")

            with open(output_file, "w", encoding="utf-8") as f:
                multiple_pages = len(result.pages) > 1
                for page in result.pages:
                    if multiple_pages:
                        f.write(f"---Page {page.page} Start---\n\n")
                    f.write(f"{page.content}\n\n")
                    if multiple_pages:
                        f.write(f"---Page {page.page} End---\n\n")

        return result
    except ValueError as e:
        return GPTParseOutput(
            file_path=os.path.abspath(file_path),
            provider=provider,
            model=model,
            completion_time=0,
            input_tokens=0,
            output_tokens=0,
            pages=[],
            error=str(e),
        )
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return GPTParseOutput(
            file_path=os.path.abspath(file_path),
            provider=provider,
            model=model,
            completion_time=0,
            input_tokens=0,
            output_tokens=0,
            pages=[],
            error=f"An unexpected error occurred: {str(e)}",
        )
