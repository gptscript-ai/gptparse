import click
from .modes.vision import vision as vision_function
from .config import get_config, set_config, print_config
import re
import os

PROVIDER_MODELS = {
    "openai": {"default": "gpt-4o", "options": ["gpt-4o", "gpt-4o-mini"]},
    "anthropic": {
        "default": "claude-3-5-sonnet-20240620",
        "options": [
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ],
    },
    "google": {
        "default": "gemini-1.5-pro-002",
        "options": [
            "gemini-1.5-pro-002",
            "gemini-1.5-flash-002",
            "gemini-1.5-flash-8b",
        ],
    },
}


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
@click.option("--select_pages", help="Pages to process (e.g., '1,3-5,10').")
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
    """Convert PDF to Markdown using vision language models."""
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
            return

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
    except ValueError as e:
        click.echo(click.style(f"Error: {str(e)}", fg="red"))
        return

    if output_file:
        click.echo(f"Output saved to {output_file}")
    else:
        multiple_pages = len(result.pages) > 1
        for page in result.pages:
            if multiple_pages:
                click.echo(
                    click.style(f"---Page {page.page} Start---", fg="yellow", bold=True)
                )
            click.echo(pretty_print_markdown(page.content))
            if multiple_pages:
                click.echo(
                    click.style(f"---Page {page.page} End---", fg="yellow", bold=True)
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


if __name__ == "__main__":
    main()
