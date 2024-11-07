import logging
from typing import Optional
from ..config import setup_logging
from ..outputs import GPTParseOutput
from .fast import fast
from .vision import vision

setup_logging()


def hybrid(
    concurrency: int,
    file_path: str,
    model: Optional[str] = None,
    output_file: Optional[str] = None,
    custom_system_prompt: Optional[str] = None,
    select_pages: Optional[str] = None,
    provider: str = "openai",
) -> GPTParseOutput:
    """Convert PDF to Markdown using fast mode first, then vision mode for enhancement."""
    try:
        # Step 1: Run fast mode
        fast_result = fast(
            file_path=file_path,
            select_pages=select_pages,
        )

        if fast_result.error:
            raise Exception(f"Fast mode error: {fast_result.error}")

        # Step 2: Prepare enhanced system prompt with fast mode results
        enhanced_prompt = custom_system_prompt or (
            "Convert the content of the image into markdown format, using the provided OCR text as a reference. It has been extracted using pymupdf. Consider that any text it extracted may be incomplete but the text it does extract is accurate."
            "You will not add any of your own commentary to your response. Consider the following:\n\n"
            "- **Tables:** Verify and correct tables in markdown format. Ensure all columns and rows match the image exactly.\n"
            "- **Lists:** Verify and correct markdown lists, maintaining the original structure (ordered or unordered).\n"
            "- **Images:** If the image contains other images, verify their descriptions within `<image></image>` tags.\n\n"
            "# Steps\n\n"
            "1. **Compare OCR and Image:**\n"
            "   - Review the provided OCR text against the image\n"
            "   - Identify any discrepancies or errors\n"
            "   - Pay special attention to numbers, special characters, and formatting\n\n"
            "2. **Enhance and Correct:**\n"
            "   - Fix any OCR errors found\n"
            "   - Ensure proper markdown syntax\n"
            "   - Maintain table structure and alignment\n"
            "   - Preserve list formatting and hierarchy\n\n"
            "3. **Verify Final Output:**\n"
            "   - Ensure all content matches the image\n"
            "   - Confirm proper markdown formatting\n"
            "   - Check structural elements (tables, lists, etc.)\n\n"
            "   - Use JSON instead of tables if the table columns are hard to format"
            "# Reference OCR Text\n\n"
        )

        # Add OCR text to prompt for each page
        for page in fast_result.pages:
            enhanced_prompt += f"\n---Page {page.page}---\n{page.content}\n"

        # Step 3: Run vision mode with enhanced prompt and prediction if using GPT-4o
        prediction = None
        if provider == "openai" and model and model.startswith("gpt-4o"):
            # Use the fast mode results as prediction for GPT-4o
            prediction = {
                "type": "content",
                "content": "\n".join(page.content for page in fast_result.pages),
            }

        vision_result = vision(
            concurrency=concurrency,
            file_path=file_path,
            model=model,
            output_file=output_file,
            custom_system_prompt=enhanced_prompt,
            select_pages=select_pages,
            provider=provider,
            prediction=prediction,  # Pass prediction to vision mode
        )

        return vision_result

    except Exception as e:
        logging.error(f"Error in hybrid mode: {str(e)}")
        return GPTParseOutput(
            file_path=file_path,
            provider=provider,
            model=model,
            completion_time=0,
            input_tokens=0,
            output_tokens=0,
            pages=[],
            error=str(e),
        )
