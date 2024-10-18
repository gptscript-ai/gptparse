from click.testing import CliRunner
from gptparse.cli import main


def test_main_command():
    runner = CliRunner()
    result = runner.invoke(main)
    assert result.exit_code == 0
    assert "GPTParse: Convert PDF documents to Markdown" in result.output


def test_vision_command_help():
    runner = CliRunner()
    result = runner.invoke(main, ["vision", "--help"])
    assert result.exit_code == 0
    assert (
        "Convert PDF to Markdown using OCR and vision language models" in result.output
    )
