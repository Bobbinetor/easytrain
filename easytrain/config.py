"""Configuration settings for EasyTrain"""

from typing import Optional
from pydantic import BaseModel, Field


class OllamaConfig(BaseModel):
    """Configuration for Ollama client"""
    host: str = Field(default="http://localhost:11434", description="Ollama server host")
    model: str = Field(default="gemma2:12b", description="Model name to use")
    timeout: int = Field(default=3600, description="Request timeout in seconds (1 hour)")


class ChunkingConfig(BaseModel):
    """Configuration for universal document chunking"""
    chunk_size: int = Field(default=2000, description="Target chunk size in characters")
    overlap_size: int = Field(default=200, description="Overlap between chunks in characters")
    min_chunk_size: int = Field(default=500, description="Minimum chunk size in characters")
    max_chunk_size: int = Field(default=4000, description="Maximum chunk size in characters")
    
    # Backward compatibility (will be converted to character-based)
    max_rows_per_chunk: int = Field(default=100, description="Legacy: Maximum rows per chunk (converted to chars)")
    overlap_rows: int = Field(default=5, description="Legacy: Number of overlapping rows (converted to chars)")
    max_text_length: int = Field(default=4000, description="Legacy: Maximum text length per chunk")
    preserve_structure: bool = Field(default=True, description="Try to preserve document structure during chunking")


class ConversionConfig(BaseModel):
    """Main configuration for the conversion process"""
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    output_format: str = Field(default="json", description="Output format: json or jsonl")
    system_prompt: str = Field(
        default="You are an expert knowledge extractor that creates high-quality training data for LLMs. Extract generalizable knowledge, concepts, and expertise from documents to create valuable question-answer pairs that would improve an LLM's knowledge base. Focus on facts, principles, procedures, and insights that can be applied beyond the specific document. Always return valid JSON arrays with no additional text or explanations.",
        description="System prompt for the LLM"
    )
    
    # Document-specific prompts
    tabular_prompt: str = Field(
        default="""Extract generalizable knowledge from this tabular data to create training examples for an LLM:

{document_data}

Instructions:
- Transform specific data points into generalizable knowledge and principles
- Create questions that teach concepts, patterns, relationships, and domain expertise
- Focus on "how to", "what is", "why does", "when should" questions that apply beyond this specific data
- Extract domain knowledge, best practices, methodologies, and expert insights
- Create educational content that would be valuable for someone learning this field
- Examples: If data shows sales patterns, create questions about sales principles; if it's financial data, create questions about financial concepts
- Format: {{"instruction": "knowledge-based question or learning task", "input": "context or scenario if needed, or empty string", "output": "expert knowledge, explanation, or principle"}}

Generate knowledge that would be valuable in a general knowledge base. Return only a JSON array with no additional text.""",
        description="Prompt template for tabular data (CSV, Excel tables)"
    )
    
    text_prompt: str = Field(
        default="""Extract valuable knowledge and expertise from this text to create training data for an LLM:

{document_data}

Instructions:
- Focus on extracting transferable knowledge, principles, methodologies, and expert insights
- Create questions that teach fundamental concepts, best practices, and domain expertise
- Transform specific examples into general principles and learning opportunities
- Generate "how-to" guides, explanations of concepts, and problem-solving approaches
- Extract definitions, frameworks, methodologies, and expert reasoning
- Create questions about cause-and-effect relationships, decision-making processes, and expert judgment
- Focus on knowledge that would be valuable for someone learning this domain or skill
- Format: {{"instruction": "knowledge-seeking question or learning task", "input": "context or scenario if needed, or empty string", "output": "expert knowledge, methodology, or principle with explanation"}}

Prioritize knowledge extraction over document-specific details. Return only a JSON array.""",
        description="Prompt template for text documents"
    )
    
    presentation_prompt: str = Field(
        default="""Extract educational knowledge and expertise from this presentation to create valuable LLM training data:

{document_data}

Instructions:
- Transform presentation content into generalizable knowledge, concepts, and expertise
- Create learning-focused questions that teach domain knowledge, methodologies, and best practices
- Extract key frameworks, processes, strategies, and expert insights presented
- Focus on "how to" implement concepts, "what are" the principles, "why do" experts recommend approaches
- Create questions about decision-making processes, evaluation criteria, and strategic thinking
- Transform slide content into educational material that builds expertise in the subject area
- Generate knowledge that would help someone become proficient in this domain
- Format: {{"instruction": "knowledge-building question or learning objective", "input": "scenario or context if needed, or empty string", "output": "expert knowledge, methodology, or strategic insight with explanation"}}

Prioritize extracting expertise and teachable knowledge over presentation-specific details. Return only a JSON array.""",
        description="Prompt template for presentations"
    )
    
    pdf_prompt: str = Field(
        default="""Extract valuable knowledge and expertise from this PDF content to create high-quality LLM training data:

{document_data}

Instructions:
- Transform document content into generalizable knowledge, principles, and domain expertise
- For tabular data: extract patterns, relationships, and domain insights rather than specific data points
- For text content: focus on methodologies, frameworks, expert reasoning, and transferable knowledge
- Create questions that teach fundamental concepts, problem-solving approaches, and best practices
- Extract definitions, procedures, decision-making frameworks, and expert insights
- Focus on "how to" apply knowledge, "what are" the key principles, "when to" use different approaches
- Generate knowledge that would be valuable for building expertise in relevant domains
- Create learning-oriented content that goes beyond document-specific information
- Format: {{"instruction": "knowledge-focused question or learning task", "input": "scenario or context if needed, or empty string", "output": "expert knowledge, principle, or methodology with clear explanation"}}

Prioritize knowledge extraction and expertise building over document summarization. Return only a JSON array.""",
        description="Prompt template for PDF documents"
    )
    
    # Fallback prompt
    user_prompt_template: str = Field(
        default="""Extract valuable knowledge and expertise from this document to create high-quality LLM training data:

Document Type: {document_type}
Content:
{document_data}

Requirements:
- Transform document content into generalizable knowledge, principles, and domain expertise
- Create learning-focused questions that build knowledge and expertise rather than test document recall
- Focus on extracting methodologies, frameworks, best practices, and expert insights
- Generate questions about "how to" apply concepts, "what are" key principles, "why do" experts recommend approaches
- Use the exact schema: {{"instruction": "knowledge-building question or learning task", "input": "scenario or context if needed, or empty string", "output": "expert knowledge, principle, or methodology with explanation"}}
- Ensure proper JSON formatting
- Return only the JSON array, no additional text
- Prioritize knowledge extraction over document-specific details""",
        description="Fallback template for unknown document types"
    )