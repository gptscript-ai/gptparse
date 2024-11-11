# GPTParse

GPTParse is a powerful and versatile document parser designed specifically for Retrieval-Augmented Generation (RAG) systems. It enables seamless conversion of PDF documents and images into Markdown format using either advanced vision language models (VLMs) or fast local processing, facilitating easy integration into text-based workflows and applications.

With GPTParse, you can:

- Convert complex PDFs and images, including those with tables, lists, and embedded images, into well-structured Markdown.
- Choose between AI-powered processing (using OpenAI, Anthropic, or Google) or fast local processing.
- Use GPTParse as a Python library or via a command-line interface (CLI), offering flexibility in how you integrate it into your projects.

It's as simple as:

```bash
# Convert a PDF using Vision Language Models
gptparse vision example.pdf --output_file output.md

# Convert a PDF using fast local processing (no VLM or internet connection required)
gptparse fast example.pdf --output_file output.md

# Convert using hybrid mode (combines fast and vision for better results)
gptparse hybrid example.pdf --output_file output.md

# Convert an image
gptparse vision screenshot.png --output_file output.md
```

## Features

- **Convert PDFs and Images to Markdown**: Transform PDF documents and image files (PNG, JPG, JPEG) into Markdown format, preserving the structure and content.
- **Multiple Parsing Methods**: Choose between using Vision Language Models (VLMs) for high-fidelity conversion, fast local processing for quick results, hybrid mode for enhanced accuracy, or OCR mode for direct text extraction.
- **Support for Multiple AI Providers**: Seamlessly integrate with OpenAI, Anthropic, and Google AI models, selecting the one that best fits your needs.
- **Python Library and CLI Application**: Use GPTParse within your Python applications or interact with it through the command line.
- **Customizable Processing Options**: Configure concurrency levels, select specific pages to process, and customize system prompts to tailor the output.
- **Page Selection**: Process entire documents or specify individual pages or ranges of pages.
- **Detailed Statistics**: Optionally display detailed processing statistics, including token usage and processing times.

## Table of Contents

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Setting Up Environment Variables](#setting-up-environment-variables)
  - [Configuration](#configuration)
  - [Using GPTParse as a Python Package](#using-gptparse-as-a-python-package)
  - [Using GPTParse via the CLI](#using-gptparse-via-the-cli)
    - [Vision Mode](#vision-mode)
    - [Fast Mode](#fast-mode)
    - [Hybrid Mode](#hybrid-mode)
    - [OCR Mode](#ocr-mode)
- [Available Models and Providers](#available-models-and-providers)
  - [OpenAI Models](#openai-models)
  - [Anthropic Models](#anthropic-models)
  - [Google Models](#google-models)
- [Examples](#examples)
  - [Processing Specific Pages](#processing-specific-pages)
  - [Custom System Prompt](#custom-system-prompt)
  - [Displaying Statistics](#displaying-statistics)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

Install GPTParse using pip:

```bash
pip install gptparse
```

### Prerequisites

Ensure you have the following installed:

- **Python 3.9** or higher
- **Poppler**: For PDF to image conversion

#### Installing Poppler

Poppler is the underlying project that handles PDF processing. You can check if you already have it installed by running `pdftoppm -h` in your terminal/command prompt.

- **Ubuntu/Debian**:

  ```bash
  sudo apt-get install poppler-utils
  ```

- **Arch Linux**:

  ```bash
  sudo pacman -S poppler
  ```

- **macOS (with Homebrew)**:

  ```bash
  brew install poppler
  ```

- **Windows**:

  1. Download the latest poppler package from [oschwartz10612's version](https://github.com/oschwartz10612/poppler-windows/releases/), which is the most up-to-date.
  2. Extract the downloaded package and move the extracted directory to your desired location.
  3. Add the `bin/` directory from the extracted folder to your [system PATH](https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/).
  4. Verify the installation by opening a new command prompt and running `pdftoppm -h`.

After installing Poppler, you should be ready to use GPTParse.

## Quick Start

Here's how you can quickly get started with GPTParse:

```bash
# Set your API key
export OPENAI_API_KEY="your-openai-api-key"

# Convert a PDF to Markdown using Vision Language Models
gptparse vision example.pdf --output_file output.md

# Convert a PDF to Markdown using fast local processing (no VLM or internet connection required)
gptparse fast example.pdf --output_file output.md

# Convert using hybrid mode (combines fast and vision for better results)
gptparse hybrid example.pdf --output_file output.md

# Convert using OCR mode (direct text extraction)
gptparse ocr example.pdf --output_file output.md
```

## Usage

### Setting Up Environment Variables

Before using GPTParse, set up the API keys for the AI providers you plan to use by setting the appropriate environment variables:

- **OpenAI**:

  ```bash
  export OPENAI_API_KEY="your-openai-api-key"
  ```

- **Anthropic**:

  ```bash
  export ANTHROPIC_API_KEY="your-anthropic-api-key"
  ```

- **Google**:

  ```bash
  export GOOGLE_API_KEY="your-google-api-key"
  ```

You can set these variables in your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) or include them in your Python script before importing GPTParse.

> **Note**: Keep your API keys secure and do not expose them in code repositories.

### Configuration

GPTParse allows you to set default configurations for ease of use. Use the `configure` command to set default values for the AI provider, model, and concurrency:

```bash
gptparse configure
```

You will be prompted to enter the desired provider, model, and concurrency level. The configuration is saved in `~/.gptparse_config.json`.

Example:

```bash
$ gptparse configure
GPTParse Configuration
Enter new values or press Enter to keep the current values.
Current values are shown in [brackets].

AI Provider [openai]: anthropic
Default Model for anthropic [claude-3-5-sonnet-latest]: claude-3-opus-latest
Default Concurrency [10]: 5
Configuration updated successfully.

Current configuration:
  provider: anthropic
  model: claude-3-opus-latest
  concurrency: 5
```

### Using GPTParse as a Python Package

Below is an example of how to use GPTParse in your Python code:

```python
import os

# For AI-powered vision processing
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
from gptparse.modes.vision import vision
from gptparse.modes.fast import fast
from gptparse.modes.hybrid import hybrid

# Using vision mode
vision_result = vision(
    concurrency=10,
    file_path="example.pdf",
    model="gpt-4o",
    output_file="output.md",
    custom_system_prompt=None,
    select_pages=None,
    provider="openai",
)

# Using fast mode (no AI required)
fast_result = fast(
    file_path="example.pdf",
    output_file="output.md",
    select_pages=None,
)

# Using hybrid mode (combines fast and vision)
hybrid_result = hybrid(
    concurrency=10,
    file_path="example.pdf",
    model="gpt-4o",
    output_file="output.md",
    custom_system_prompt=None,
    select_pages=None,
    provider="openai",
)
```

### Using GPTParse via the CLI

When using the command-line interface, you have four modes available:

1. **Vision Mode** - Uses AI models for high-quality conversion:

```bash
export OPENAI_API_KEY="your-openai-api-key"
gptparse vision example.pdf --output_file output.md --provider openai
```

2. **Fast Mode** - Uses local processing for quick conversion (no AI required):

```bash
gptparse fast example.pdf --output_file output.md
```

3. **Hybrid Mode** - Combines fast and vision modes for enhanced results:

```bash
export OPENAI_API_KEY="your-openai-api-key"
gptparse hybrid example.pdf --output_file output.md --provider openai
```

4. **OCR Mode** - Uses direct OCR processing for text extraction:

```bash
gptparse ocr example.pdf --output_file output.md
```

- `--output_file`: Output file name (must have a `.md` or `.txt` extension).
- `--abort-on-error`: Stop processing if an error occurs (optional).

#### Vision Mode Options

- `--concurrency`: Number of concurrent processes (default: value set in configuration or 10).
- `--model`: Vision language model to use (overrides configured default).
- `--output_file`: Output file name (must have a `.md` or `.txt` extension).
- `--custom_system_prompt`: Custom system prompt for the language model.
- `--select_pages`: Pages to process (e.g., `"1,3-5,10"`). Only applicable for PDF files.
- `--provider`: AI provider to use (`openai`, `anthropic`, `google`).
- `--stats`: Display detailed statistics after processing.

#### Fast Mode Options

- `--output_file`: Output file name (must have a `.md` or `.txt` extension).
- `--select_pages`: Pages to process (e.g., `"1,3-5,10"`). Only applicable for PDF files.
- `--stats`: Display basic processing statistics.

#### Hybrid Mode Options
- `--concurrency`: Number of concurrent processes (default: value set in configuration or 10).
- `--model`: Vision language model to use (overrides configured default).
- `--output_file`: Output file name (must have a `.md` or `.txt` extension).
- `--custom_system_prompt`: Custom system prompt for the language model.
- `--select_pages`: Pages to process (e.g., `"1,3-5,10"`). Only applicable for PDF files.
- `--provider`: AI provider to use (`openai`, `anthropic`, `google`).
- `--stats`: Display detailed statistics after processing.

#### OCR Mode Options

```bash
gptparse ocr example.pdf --output_file output.md
```

- `--output_file`: Output file name (must have a `.md` or `.txt` extension).
- `--abort-on-error`: Stop processing if an error occurs (optional).

## Available Models and Providers

GPTParse supports multiple models from different AI providers.

### OpenAI Models

- `gpt-4o` (Default)
- `gpt-4o-mini`

### Anthropic Models

- `claude-3-5-sonnet-latest` (Default)
- `claude-3-opus-latest`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

### Google Models

- `gemini-1.5-pro-002` (Default)
- `gemini-1.5-flash-002`
- `gemini-1.5-flash-8b`

To list available models for a provider in your code, you can use:

```python
from gptparse.models.model_interface import list_available_models

# List models for a specific provider
models = list_available_models(provider='openai')
print("OpenAI models:", models)

# List all available models from all providers
all_models = list_available_models()
print("All available models:", all_models)
```

## Examples

### Processing Specific Pages

To process only specific pages from a PDF document, use the `--select_pages` option:

```bash
gptparse vision example.pdf --select_pages "2,4,6-8"
```

This command will process pages 2, 4, 6, 7, and 8 of `example.pdf`.

### Custom System Prompt

Provide a custom system prompt to influence the model's output:

```bash
gptparse vision example.pdf --custom_system_prompt "Please extract all text in bullet points."
```

### Displaying Statistics

To display detailed processing statistics, use the `--stats` flag:

```bash
gptparse vision example.pdf --stats
```

Sample output:

```
Detailed Statistics:
File Path: example.pdf
Completion Time: 12.34 seconds
Total Pages Processed: 5
Total Input Tokens: 2500
Total Output Tokens: 3000
Total Tokens: 5500
Average Tokens per Page: 1100.00

Page-wise Statistics:
  Page 1: 600 tokens
  Page 2: 500 tokens
  Page 3: 700 tokens
  Page 4: 800 tokens
  Page 5: 400 tokens
```

### Processing Images

To process an image file:

```bash
# Process a PNG file
gptparse vision screenshot.png --output_file output.md

# Process a JPG file
gptparse vision photo.jpg --output_file output.md
```

Supported image formats:

- PNG
- JPG/JPEG

### Processing with OCR

To process a file using direct OCR:

```bash
# Process a PDF file with OCR
gptparse ocr document.pdf --output_file output.md

# Process an image with OCR
gptparse ocr scan.png --output_file output.md

# Process with abort-on-error flag
gptparse ocr document.pdf --output_file output.md --abort-on-error
```

The OCR mode supports:
- PDF documents
- PNG images
- JPG/JPEG images

## Contributing

Contributions are welcome! If you'd like to contribute to GPTParse, please follow these steps:

1. **Fork the repository** on GitHub.
2. **Create a new branch** for your feature or bugfix.
3. **Make your changes** and ensure tests pass.
4. **Submit a pull request** with a clear description of your changes.

Please ensure that your code adheres to the existing style conventions and passes all tests.

## License

GPTParse is licensed under the Apache-2.0 License. See [LICENSE](LICENSE) for more information.
