"""Configuration settings for EasyTrain"""

from typing import Optional
from pydantic import BaseModel, Field


class OllamaConfig(BaseModel):
    """Configuration for Ollama client"""
    host: str = Field(default="http://localhost:11434", description="Ollama server host")
    model: str = Field(default="gemma2:12b", description="Model name to use")
    timeout: int = Field(default=300, description="Request timeout in seconds")


class ChunkingConfig(BaseModel):
    """Configuration for document chunking"""
    max_rows_per_chunk: int = Field(default=100, description="Maximum rows per chunk")
    overlap_rows: int = Field(default=5, description="Number of overlapping rows between chunks")
    max_text_length: int = Field(default=4000, description="Maximum text length per chunk for text documents")
    preserve_structure: bool = Field(default=True, description="Try to preserve document structure during chunking")


class ConversionConfig(BaseModel):
    """Main configuration for the conversion process"""
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    output_format: str = Field(default="jsonl", description="Output format: json or jsonl")
    
    # Document-specific prompts
    tabular_prompt: str = Field(
        default="""Convert this tabular data to JSON format for LLM training:

{document_data}

Instructions:
- Each row should become a training example with question-answer format
- Extract meaningful relationships between columns
- Create diverse question types: direct questions, comparison questions, analytical questions
- Use column headers to form natural questions
- Preserve all original data values accurately
- Format: {{"instruction": "question about the data", "response": "answer based on row data", "category": "relevant category if available"}}

Return only a JSON array with no additional text.""",
        description="Prompt template for tabular data (CSV, Excel tables)"
    )
    
    text_prompt: str = Field(
        default="""Convert this text content to JSON format for LLM training:

{document_data}

Instructions:
- Extract key concepts, facts, and explanations from the text
- Create question-answer pairs that test understanding of the content
- Generate different types of questions: factual, conceptual, application-based
- Break down complex topics into digestible Q&A pairs
- Include definitions, explanations, and examples where present
- Create instructional pairs for step-by-step processes
- Format: {{"instruction": "question or instruction", "response": "detailed answer or explanation", "topic": "subject area"}}

Ensure comprehensive coverage of the text content. Return only a JSON array.""",
        description="Prompt template for text documents"
    )
    
    presentation_prompt: str = Field(
        default="""Convert this presentation content to JSON format for LLM training:

{document_data}

Instructions:
- Use slide titles as topics/subjects for questions
- Extract key points and main ideas from slide content
- Create questions about slide relationships and flow
- Generate summary questions for each slide
- Include questions about slide structure and organization
- Create both specific detail questions and broader concept questions
- Format: {{"instruction": "question about slide content or presentation topic", "response": "answer based on slide information", "slide_context": "relevant slide title or number"}}

Focus on the educational value of the presentation content. Return only a JSON array.""",
        description="Prompt template for presentations"
    )
    
    pdf_prompt: str = Field(
        default="""Convert this PDF content to JSON format for LLM training:

{document_data}

Instructions:
- If content contains tables, treat each row as potential training data
- For text sections, extract key information and create Q&A pairs
- Focus on factual information, procedures, and explanations
- Create questions about document structure and organization
- Extract definitions, processes, and step-by-step instructions
- Include questions about relationships between different sections
- For mixed content, create varied question types appropriate to each section
- Format: {{"instruction": "question or instruction", "response": "detailed answer", "content_type": "table/text/mixed", "source_section": "relevant section if identifiable"}}

Maximize the educational value from the document content. Return only a JSON array.""",
        description="Prompt template for PDF documents"
    )
    
    # Fallback prompt
    user_prompt_template: str = Field(
        default="""Convert this document data to JSON format for LLM training:

Document Type: {document_type}
Content:
{document_data}

Requirements:
- Create meaningful training examples appropriate for the document type
- Use clear field names (e.g., "instruction", "response", "input", "output")
- Ensure proper JSON formatting
- Return only the JSON array, no additional text
- For tabular data: each row becomes a training example
- For text: create instruction-response or question-answer pairs
- For presentations: use titles and content meaningfully""",
        description="Fallback template for unknown document types"
    )