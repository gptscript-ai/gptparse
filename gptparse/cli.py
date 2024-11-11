import click
from .modes.vision import vision as vision_function
from .config import get_config, set_config, print_config
from gptparse.models.model_interface import PROVIDER_MODELS
import re
import os
import sys
import textwrap


def pretty_print_markdown(markdown_content):
    """
    Format markdown content for better readability in the terminal.
    """
    # Replace markdown headers with colored and bold text
    markdown_content = re.sub(
        r"^(#+)\s*(.*?)$",
        lambda m: click.style(f"{m.group(2)}", fg="cyan", bold=True),
        markdown_content,
        flags=re.MULTILINE,
    )

    # Replace bold text
    markdown_content = re.sub(
        r"\*\*(.*?)\*\*",
        lambda m: click.style(f"{m.group(1)}", bold=True),
        markdown_content,
    )

    # Replace italic text
    markdown_content = re.sub(
        r"\*(.*?)\*",
        lambda m: click.style(f"{m.group(1)}", italic=True),
        markdown_content,
    )

    # Replace code blocks
    markdown_content = re.sub(
        r"```(.*?)```",
        lambda m: click.style(f"{m.group(1)}", fg="green"),
        markdown_content,
        flags=re.DOTALL,
    )

    return markdown_content


def format_authentication_error(error_message, provider):
    return textwrap.dedent(
        f"""
        {error_message}

        Please check the following:
        - Ensure your API key for {provider.capitalize()} is correct.
        - Verify that the API key is properly set in your environment variables.
        - Check your account status and API key validity.

        API Key Management:
        - OpenAI: https://platform.openai.com/account/api-keys
        - Anthropic: https://console.anthropic.com/settings/keys
        - Google (Gemini): https://makersuite.google.com/app/apikey
    """
    ).strip()


@click.group()
def main():
    """GPTParse: Convert PDF documents to Markdown using OCR and vision language models."""
    pass


@main.command()
@click.option("--concurrency", default=10, help="Number of concurrent processes.")
@click.argument("file_path", type=click.Path(exists=True, resolve_path=True))
@click.option("--model", help="Vision language model to use.")
@click.option(
    "--output_file",
    help="Output file name (with .md or .txt extension). If not specified, output will be printed.",
)
@click.option(
    "--custom_system_prompt", help="Custom system prompt for the language model."
)
@click.option(
    "--select_pages",
    help="Pages to process (e.g., '1,3-5,10'). Only applicable for PDFs.",
)
@click.option("--provider", help="AI provider to use (openai, anthropic, or google).")
@click.option(
    "--stats", is_flag=True, help="Display detailed statistics after processing."
)
def vision(
    concurrency,
    file_path,
    model,
    output_file,
    custom_system_prompt,
    select_pages,
    provider,
    stats,
):
    """Convert PDF or image files to Markdown using vision language models."""
    config = get_config()
    provider = provider or config.get("provider", "openai")
    model = model or config.get("model")

    if output_file:
        _, ext = os.path.splitext(output_file)
        if ext.lower() not in (".md", ".txt"):
            click.echo(
                click.style(
                    "Error: Output file must have a .md or .txt extension", fg="red"
                )
            )
            sys.exit(1)

    _, input_ext = os.path.splitext(file_path)
    input_ext = input_ext.lower()

    if input_ext not in (".pdf", ".png", ".jpg", ".jpeg"):
        click.echo(
            click.style(
                "Error: Input file must be a PDF or image file (PNG, JPG)", fg="red"
            )
        )
        sys.exit(1)

    try:
        result = vision_function(
            concurrency=concurrency,
            file_path=file_path,
            model=model,
            output_file=output_file,
            custom_system_prompt=custom_system_prompt,
            select_pages=select_pages,
            provider=provider,
        )

        if result.error:
            raise Exception(result.error)

        if output_file:
            click.echo(f"Output saved to {output_file}")
        else:
            multiple_pages = len(result.pages) > 1
            for page in result.pages:
                if multiple_pages:
                    click.echo(
                        click.style(
                            f"---Page {page.page} Start---", fg="cyan", bold=True
                        )
                    )
                click.echo(pretty_print_markdown(page.content))
                if multiple_pages:
                    click.echo(
                        click.style(f"---Page {page.page} End---", fg="cyan", bold=True)
                    )
                click.echo("\n")

        if stats:
            click.echo(click.style("Detailed Statistics:", fg="blue", bold=True))
            click.echo(f"File Path: {file_path}")
            click.echo(f"Completion Time: {result.completion_time:.2f} seconds")
            click.echo(f"Total Pages Processed: {len(result.pages)}")
            click.echo(f"Total Input Tokens: {result.input_tokens}")
            click.echo(f"Total Output Tokens: {result.output_tokens}")
            click.echo(f"Total Tokens: {result.input_tokens + result.output_tokens}")

            avg_tokens_per_page = (
                (result.input_tokens + result.output_tokens) / len(result.pages)
                if result.pages
                else 0
            )
            click.echo(f"Average Tokens per Page: {avg_tokens_per_page:.2f}")

            click.echo("\nPage-wise Statistics:")
            for page in result.pages:
                click.echo(f"  Page {page.page}: {page.output_tokens} tokens")

    except Exception as e:
        error_message = str(e)
        if "authentication error" in error_message.lower():
            error_message = format_authentication_error(error_message, provider)

        click.echo(click.style(f"Error: {error_message}", fg="red"))
        sys.exit(1)


@main.command()
def configure():
    """Configure default settings for GPTParse."""
    config = get_config()

    print("GPTParse Configuration")
    print("Enter new values or press Enter to keep the current values.")
    print("Current values are shown in [brackets].")
    print()

    providers = list(PROVIDER_MODELS.keys())
    provider = click.prompt(
        "AI Provider",
        default=config.get("provider", "openai"),
        type=click.Choice(providers),
        show_default=True,
        show_choices=True,
    )

    default_model = PROVIDER_MODELS[provider]["default"]
    model_options = PROVIDER_MODELS[provider]["options"]
    model = click.prompt(
        f"Default Model for {provider}",
        default=config.get("model", default_model),
        type=click.Choice(model_options),
        show_default=True,
        show_choices=True,
    )

    concurrency = click.prompt(
        "Default Concurrency",
        default=config.get("concurrency", 10),
        type=int,
        show_default=True,
    )

    new_config = {"provider": provider, "model": model, "concurrency": concurrency}

    set_config(new_config)
    click.echo("Configuration updated successfully.")
    print()
    print_config()


@main.command()
@click.argument("file_path", type=click.Path(exists=True, resolve_path=True))
@click.option(
    "--output_file",
    help="Output file name (with .md or .txt extension). If not specified, output will be printed.",
)
@click.option("--select_pages", help="Pages to process (e.g., '1,3-5,10')")
@click.option(
    "--stats", is_flag=True, help="Display basic statistics after processing."
)
def fast(
    file_path,
    output_file,
    select_pages,
    stats,
):
    """Convert PDF files to Markdown using fast local processing (no AI)."""
    if output_file:
        _, ext = os.path.splitext(output_file)
        if ext.lower() not in (".md", ".txt"):
            click.echo(
                click.style(
                    "Error: Output file must have a .md or .txt extension", fg="red"
                )
            )
            sys.exit(1)

    _, input_ext = os.path.splitext(file_path)
    input_ext = input_ext.lower()

    if input_ext not in (".pdf", ".txt"):
        click.echo(
            click.style("Error: Input file must be a PDF or text file", fg="red")
        )
        sys.exit(1)

    try:
        from .modes.fast import fast as fast_function

        result = fast_function(
            file_path=file_path,
            output_file=output_file,
            select_pages=select_pages,
        )

        if result.error:
            raise Exception(result.error)

        if output_file:
            click.echo(f"Output saved to {output_file}")
        else:
            multiple_pages = len(result.pages) > 1
            for page in result.pages:
                if multiple_pages:
                    click.echo(
                        click.style(
                            f"---Page {page.page} Start---", fg="cyan", bold=True
                        )
                    )
                click.echo(pretty_print_markdown(page.content))
                if multiple_pages:
                    click.echo(
                        click.style(f"---Page {page.page} End---", fg="cyan", bold=True)
                    )
                click.echo("\n")

        if stats:
            click.echo(click.style("Processing Statistics:", fg="blue", bold=True))
            click.echo(f"File Path: {file_path}")
            click.echo(f"Completion Time: {result.completion_time:.2f} seconds")
            click.echo(f"Total Pages Processed: {len(result.pages)}")

    except Exception as e:
        click.echo(click.style(f"Error: {str(e)}", fg="red"))
        sys.exit(1)


@main.command()
@click.option("--concurrency", default=10, help="Number of concurrent processes.")
@click.argument("file_path", type=click.Path(exists=True, resolve_path=True))
@click.option("--model", help="Vision language model to use.")
@click.option(
    "--output_file",
    help="Output file name (with .md or .txt extension). If not specified, output will be printed.",
)
@click.option(
    "--custom_system_prompt", help="Custom system prompt for the language model."
)
@click.option(
    "--select_pages",
    help="Pages to process (e.g., '1,3-5,10'). Only applicable for PDFs.",
)
@click.option("--provider", help="AI provider to use (openai, anthropic, or google).")
@click.option(
    "--stats", is_flag=True, help="Display detailed statistics after processing."
)
def hybrid(
    concurrency,
    file_path,
    model,
    output_file,
    custom_system_prompt,
    select_pages,
    provider,
    stats,
):
    """Convert PDF files using fast local processing followed by AI enhancement."""
    if output_file:
        _, ext = os.path.splitext(output_file)
        if ext.lower() not in (".md", ".txt"):
            click.echo(
                click.style(
                    "Error: Output file must have a .md or .txt extension", fg="red"
                )
            )
            sys.exit(1)

    _, input_ext = os.path.splitext(file_path)
    input_ext = input_ext.lower()

    if input_ext not in (".pdf", ".txt"):
        click.echo(
            click.style("Error: Input file must be a PDF or text file", fg="red")
        )
        sys.exit(1)

    try:
        from .modes.hybrid import hybrid as hybrid_function

        result = hybrid_function(
            concurrency=concurrency,
            file_path=file_path,
            model=model,
            output_file=output_file,
            custom_system_prompt=custom_system_prompt,
            select_pages=select_pages,
            provider=provider,
        )

        if result.error:
            raise Exception(result.error)

        if output_file:
            click.echo(f"Output saved to {output_file}")
        else:
            multiple_pages = len(result.pages) > 1
            for page in result.pages:
                if multiple_pages:
                    click.echo(
                        click.style(
                            f"---Page {page.page} Start---", fg="cyan", bold=True
                        )
                    )
                click.echo(pretty_print_markdown(page.content))
                if multiple_pages:
                    click.echo(
                        click.style(f"---Page {page.page} End---", fg="cyan", bold=True)
                    )
                click.echo("\n")

        if stats:
            click.echo(click.style("Processing Statistics:", fg="blue", bold=True))
            click.echo(f"File Path: {file_path}")
            click.echo(f"Completion Time: {result.completion_time:.2f} seconds")
            click.echo(f"Total Pages Processed: {len(result.pages)}")
            click.echo(f"Total Input Tokens: {result.input_tokens}")
            click.echo(f"Total Output Tokens: {result.output_tokens}")
            click.echo(f"Total Tokens: {result.input_tokens + result.output_tokens}")

    except Exception as e:
        error_message = str(e)
        if "authentication error" in error_message.lower():
            error_message = format_authentication_error(error_message, provider)
        click.echo(click.style(f"Error: {error_message}", fg="red"))
        sys.exit(1)


@main.command()
@click.argument("file_path", type=click.Path(exists=True, resolve_path=True))
@click.option("--output_file", help="Output file name (with .md or .txt extension)")
@click.option("--abort-on-error", is_flag=True, help="Abort on first error")
def ocr(file_path, output_file, abort_on_error):
    """Convert PDF files to text using OCR."""

    # Validate output file extension
    if output_file:
        _, ext = os.path.splitext(output_file)
        if ext.lower() not in (".md", ".txt"):
            click.echo(
                click.style(
                    "Error: Output file must have a .md or .txt extension", fg="red"
                )
            )
            sys.exit(1)

    # Validate input file
    _, input_ext = os.path.splitext(file_path)
    input_ext = input_ext.lower()
    if input_ext not in (".pdf", ".png", ".jpg", ".jpeg"):
        click.echo(
            click.style(
                "Error: Input file must be a PDF or image file (PNG, JPG)", fg="red"
            )
        )
        sys.exit(1)

    try:
        from .handlers.docling_handler import DoclingHandler

        # Create handler with file_path
        handler = DoclingHandler(file_path=file_path, abort_on_error=abort_on_error)

        result = handler.process(output_file)

        if output_file:
            click.echo(f"Output saved to {output_file}")
        else:
            click.echo(result)

    except Exception as e:
        click.echo(click.style(f"Error: {str(e)}", fg="red"))
        if abort_on_error:
            sys.exit(1)


if __name__ == "__main__":
    main()
