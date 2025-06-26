# EasyTrain

A Python suite for converting various document formats (CSV, Excel, Word, PDF, PowerPoint) to JSON format suitable for LLM training using Ollama with local models.

## Features

- **Multi-format Support**: Convert CSV, XLSX, XLS, DOCX, DOC, PDF, PPTX, PPT files
- **Intelligent Parsing**: Automatically detects and handles different document types
- **Smart Chunking**: Handles large documents with configurable chunking strategies
- **LLM Integration**: Uses Ollama for intelligent document-to-JSON conversion
- **Flexible Output**: Supports JSON and JSONL formats
- **Progress Tracking**: Real-time progress updates and detailed logging
- **CLI Interface**: Easy-to-use command line interface

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd easytrain
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Or install in development mode:
```bash
pip install -e .
```

**Note**: Some document processing features require additional system dependencies:
- For PDF processing: Install Java (required by tabula-py)
- For advanced PDF table extraction: Install OpenCV (`pip install opencv-python`)
- On macOS: You may need to install Cairo (`brew install cairo`)

## Prerequisites

- Python 3.8+
- Ollama installed and running
- A compatible model (e.g., gemma2:12b)

### Setting up Ollama

1. Install Ollama from https://ollama.com
2. Pull your desired model:
```bash
ollama pull gemma2:12b
```
3. Start Ollama server (usually starts automatically)

## Usage

### Basic Usage

Convert any supported file to JSON:
```bash
python main.py convert input.xlsx output.jsonl
python main.py convert document.pdf output.jsonl
python main.py convert presentation.pptx output.jsonl
```

### Show Supported Formats

```bash
python main.py formats
```

### Advanced Usage

```bash
python main.py convert input.docx output.jsonl \
  --model gemma2:12b \
  --chunk-size 50 \
  --overlap 3 \
  --max-text-length 4000 \
  --format jsonl \
  --verbose
```

### Check Status

Check if Ollama is running and see available models:
```bash
python main.py status
```

### Test Model

Test a specific model:
```bash
python main.py test-model gemma2:12b
```

## Configuration Options

- `--model`: Ollama model to use (default: gemma2:12b)
- `--host`: Ollama server host (default: http://localhost:11434)
- `--chunk-size`: Maximum rows per chunk (default: 100)
- `--overlap`: Overlapping rows between chunks (default: 5)
- `--max-text-length`: Maximum text length per chunk for text documents (default: 4000)
- `--format`: Output format - json or jsonl (default: jsonl)
- `--timeout`: Request timeout in seconds (default: 300)
- `--prompt-file`: Custom prompt file
- `--verbose`: Enable verbose logging

## Supported File Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| CSV | .csv | Comma-separated values |
| Excel | .xlsx, .xls | Excel spreadsheets |
| Word | .docx, .doc | Word documents |
| PDF | .pdf | Portable Document Format |
| PowerPoint | .pptx, .ppt | PowerPoint presentations |
| Text | .txt | Plain text files |

## Custom Prompts

You can provide custom prompts for the LLM conversion process:

```bash
python main.py convert input.csv output.jsonl --prompt-file custom_prompt.txt
```

The prompt file should contain the conversion instructions with `{document_data}` and `{document_type}` placeholders where the document content and type will be inserted.

## Project Structure

```
easytrain/
├── easytrain/
│   ├── __init__.py
│   ├── cli.py                # Command line interface
│   ├── config.py             # Configuration classes
│   ├── converter.py          # Main conversion logic
│   ├── universal_processor.py # Universal document processor
│   ├── file_detector.py      # File format detection
│   ├── document_parsers.py   # Parsers for different formats
│   ├── csv_processor.py      # Legacy CSV processor
│   └── ollama_client.py      # Ollama API client
├── main.py                   # Entry point
├── setup.py                 # Package setup
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## How It Works

1. **Format Detection**: Automatically detects the input file format (CSV, Excel, PDF, etc.)
2. **Document Parsing**: Uses specialized parsers to extract structured data from each format
3. **Smart Chunking**: Splits large documents into manageable chunks while preserving structure
4. **LLM Processing**: Each chunk is sent to your local Ollama model with format-specific prompts
5. **JSON Generation**: The LLM intelligently converts document data to JSON format suitable for training
6. **Output**: Results are saved as JSON or JSONL files ready for LLM training

## Examples

### CSV Input
```csv
question,answer,category
What is Python?,A programming language,Programming
How to install packages?,Use pip install,Programming
```

### PDF Document
The tool can extract text and tables from PDF files, converting them to structured training data.

### Word Document
Extracts paragraphs and tables from Word documents, creating instruction-response pairs.

### PowerPoint Presentation
Converts slide titles and content into meaningful training examples.

### Example Output JSON
```json
[
  {
    "instruction": "What is Python?",
    "output": "A programming language",
    "category": "Programming"
  },
  {
    "instruction": "How to install packages?",
    "output": "Use pip install",
    "category": "Programming"
  }
]
```

## Troubleshooting

- **Ollama not available**: Make sure Ollama is installed and running
- **Model not found**: Pull the model with `ollama pull <model-name>`
- **Unsupported file format**: Check supported formats with `python main.py formats`
- **PDF parsing issues**: Ensure Java is installed for tabula-py
- **Memory issues**: Reduce chunk size or max text length for large documents
- **Timeout errors**: Increase timeout or use a smaller model
- **Excel file errors**: Install openpyxl: `pip install openpyxl`
- **Word document errors**: Install python-docx: `pip install python-docx`

## License

MIT License