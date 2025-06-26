# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EasyTrain is a Python suite for converting various document formats (CSV, Excel, Word, PDF, PowerPoint) to JSON format suitable for LLM training using Ollama with local models. The tool intelligently parses different document types, chunks large documents appropriately, and uses a local LLM via Ollama to convert the data to training-ready JSON format.

## Key Architecture

- **Multi-Format Support**: Universal document processing with format-specific parsers
- **Intelligent Detection**: Automatic file format detection using MIME types and extensions
- **Modular Design**: Separated concerns with distinct modules for different document types
- **Configuration-Driven**: Uses Pydantic models for type-safe configuration management
- **Smart Chunking**: Handles different document types with appropriate chunking strategies
- **Error Handling**: Comprehensive error handling with logging throughout the pipeline
- **CLI Interface**: Click-based command line interface with multiple commands

## Core Modules

- `file_detector.py`: Detects file formats and provides file metadata
- `document_parsers.py`: Specialized parsers for each supported format (CSV, Excel, Word, PDF, PowerPoint)
- `universal_processor.py`: Universal document processor that routes to appropriate parsers and handles chunking
- `ollama_client.py`: Manages Ollama API interactions with error handling and model validation
- `converter.py`: Orchestrates the conversion process, manages chunks, and handles LLM responses
- `config.py`: Pydantic-based configuration classes for all components
- `cli.py`: Click-based CLI with commands for convert, status, test-model, and formats
- `csv_processor.py`: Legacy CSV-specific processor (kept for compatibility)

## Development Commands

### Setup and Installation
```bash
# Install in development mode
pip install -e .

# Install requirements
pip install -r requirements.txt
```

### Running the Tool
```bash
# Basic conversion (any supported format)
python main.py convert input.xlsx output.jsonl
python main.py convert document.pdf output.jsonl
python main.py convert presentation.pptx output.jsonl

# With custom parameters
python main.py convert input.docx output.jsonl --model gemma2:12b --chunk-size 50 --max-text-length 4000

# Check supported formats
python main.py formats

# Check Ollama status
python main.py status

# Test model
python main.py test-model gemma2:12b
```

### Testing Commands
```bash
# No formal test suite yet - manual testing via CLI commands
python main.py status  # Test Ollama connectivity
python main.py test-model <model-name>  # Test specific model
```

## Important Implementation Details

### Multi-Format Document Parsing
- Automatic format detection using MIME types with extension fallback
- Specialized parsers for each format with appropriate extraction strategies
- Handles complex documents with tables, text, and mixed content
- PDF processing includes both text extraction and table detection
- Excel processing handles multiple sheets and combines them intelligently

### Smart Chunking Strategies
- Different chunking approaches for tabular vs text-based documents
- Preserves document structure when possible (paragraphs, slides, tables)
- Configurable limits for both row count and text length
- Overlapping chunks for tabular data to maintain context
- Structure-aware chunking for presentations and documents

### Document Type Handling
- **Tabular**: CSV, Excel tables, PDF tables - row-based processing
- **Text**: Word documents, PDF text - paragraph/section-based processing  
- **Presentation**: PowerPoint - slide-based processing with title/content pairs
- Format-specific prompt templates and processing logic

### LLM Response Parsing
- Robust JSON extraction from LLM responses with fallback parsing
- Handles both JSON arrays and single objects
- Validates and logs parsing issues for debugging
- Format-aware response processing

### Error Recovery
- Continues processing even if individual chunks fail
- Graceful handling of unsupported files or parsing errors
- Detailed error logging with format-specific guidance
- Validates file format support before processing

## Code Patterns

- Use logging extensively for debugging and monitoring
- Validate inputs and system requirements before processing
- Handle errors gracefully without stopping entire conversion
- Use type hints throughout for better code maintainability
- Follow separation of concerns with single-responsibility modules