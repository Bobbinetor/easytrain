"""Command line interface for EasyTrain"""

import click
import logging
from pathlib import Path
from .config import ConversionConfig, OllamaConfig, ChunkingConfig
from .converter import DatasetConverter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option()
def cli():
    """EasyTrain - CSV to JSON Dataset Converter for LLM Training"""
    pass


@cli.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.argument('output_json', type=click.Path(path_type=Path))
@click.option('--model', default='gemma2:12b', help='Ollama model to use')
@click.option('--host', default='http://localhost:11434', help='Ollama server host')
@click.option('--chunk-size', default=100, help='Maximum rows per chunk')
@click.option('--overlap', default=5, help='Overlapping rows between chunks')
@click.option('--max-text-length', default=4000, help='Maximum text length per chunk')
@click.option('--format', 'output_format', default='jsonl', 
              type=click.Choice(['json', 'jsonl']), help='Output format')
@click.option('--timeout', default=300, help='Request timeout in seconds')
@click.option('--prompt-file', type=click.Path(exists=True, path_type=Path), 
              help='Custom prompt file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def convert(input_file: Path, 
           output_json: Path,
           model: str,
           host: str,
           chunk_size: int,
           overlap: int,
           max_text_length: int,
           output_format: str,
           timeout: int,
           prompt_file: Path,
           verbose: bool):
    """Convert any supported file to JSON dataset using Ollama LLM"""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load custom prompt if provided
    custom_prompt = None
    if prompt_file:
        try:
            custom_prompt = prompt_file.read_text(encoding='utf-8')
            logger.info(f"Loaded custom prompt from {prompt_file}")
        except Exception as e:
            logger.error(f"Failed to load prompt file: {e}")
            return
    
    # Check file format first
    from .file_detector import FileDetector
    if not FileDetector.is_supported(input_file):
        file_format = FileDetector.detect_format(input_file)
        click.echo(f"❌ Unsupported file format: {file_format.value}")
        click.echo(f"Supported formats: {', '.join(FileDetector.get_supported_formats())}")
        return
    
    # Show file info
    file_info = FileDetector.get_file_info(input_file)
    click.echo(f"📄 File: {file_info['name']} ({file_info['format'].value}, {file_info['size_mb']}MB)")
    
    # Create configuration
    config = ConversionConfig(
        ollama=OllamaConfig(
            host=host,
            model=model,
            timeout=timeout
        ),
        chunking=ChunkingConfig(
            max_rows_per_chunk=chunk_size,
            overlap_rows=overlap,
            max_text_length=max_text_length
        ),
        output_format=output_format
    )
    
    # Create converter and validate setup
    converter = DatasetConverter(config)
    validation = converter.validate_setup()
    
    if not validation["ollama_available"]:
        logger.error("Ollama server is not available. Please start Ollama first.")
        return
    
    if not validation["model_exists"]:
        logger.error(f"Model '{model}' not found. Please pull the model first:")
        logger.error(f"  ollama pull {model}")
        return
    
    logger.info(f"Converting {input_file} to {output_json}")
    logger.info(f"Using model: {model}")
    logger.info(f"Chunk size: {chunk_size}, Overlap: {overlap}, Max text length: {max_text_length}")
    
    # Perform conversion
    success = converter.convert(input_file, output_json, custom_prompt)
    
    if success:
        click.echo(f"✅ Successfully converted to {output_json}")
    else:
        click.echo("❌ Conversion failed")
        exit(1)


@cli.command()
@click.option('--host', default='http://localhost:11434', help='Ollama server host')
def status(host: str):
    """Check Ollama server status and available models"""
    
    config = OllamaConfig(host=host)
    from .ollama_client import OllamaClient
    
    client = OllamaClient(config)
    
    if not client.is_available():
        click.echo("❌ Ollama server is not available")
        return
    
    click.echo("✅ Ollama server is running")
    
    models = client.list_models()
    if models:
        click.echo("\nAvailable models:")
        for model in models:
            name = model.get("name", "Unknown")
            size = model.get("size", 0)
            size_gb = size / (1024**3) if size else 0
            click.echo(f"  - {name} ({size_gb:.1f}GB)")
    else:
        click.echo("\nNo models found")


@cli.command()
@click.argument('model_name')
@click.option('--host', default='http://localhost:11434', help='Ollama server host')
def test_model(model_name: str, host: str):
    """Test a specific model with a simple prompt"""
    
    config = OllamaConfig(host=host, model=model_name)
    from .ollama_client import OllamaClient
    
    client = OllamaClient(config)
    
    if not client.is_available():
        click.echo("❌ Ollama server is not available")
        return
    
    if not client.model_exists(model_name):
        click.echo(f"❌ Model '{model_name}' not found")
        return
    
    click.echo(f"Testing model: {model_name}")
    
    test_prompt = "Convert this data to JSON for LLM training: name,age\nJohn,25\nJane,30"
    
    try:
        response = client.generate(
            prompt=test_prompt,
            system_prompt="You are a helpful assistant that converts CSV to JSON format."
        )
        
        click.echo("\nResponse:")
        click.echo(response.get("response", "No response"))
        
    except Exception as e:
        click.echo(f"❌ Error testing model: {e}")


@cli.command()
def formats():
    """Show supported file formats"""
    from .file_detector import FileDetector
    
    click.echo("📋 Supported file formats:")
    formats = FileDetector.get_supported_formats()
    
    format_descriptions = {
        'csv': 'Comma-separated values',
        'xlsx': 'Excel spreadsheet (2007+)',
        'xls': 'Excel spreadsheet (legacy)',
        'pdf': 'Portable Document Format',
        'docx': 'Word document (2007+)',
        'doc': 'Word document (legacy)',
        'pptx': 'PowerPoint presentation (2007+)',
        'ppt': 'PowerPoint presentation (legacy)',
        'txt': 'Plain text file'
    }
    
    for fmt in formats:
        description = format_descriptions.get(fmt, 'Document file')
        click.echo(f"  • .{fmt} - {description}")


if __name__ == '__main__':
    cli()