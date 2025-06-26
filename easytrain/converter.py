"""Main converter class that orchestrates document to JSON conversion"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .config import ConversionConfig
from .universal_processor import UniversalProcessor
from .ollama_client import OllamaClient
from .file_detector import FileDetector

logger = logging.getLogger(__name__)


class DatasetConverter:
    """Main class for converting any supported document to JSON format using LLM"""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        self.universal_processor = UniversalProcessor(config.chunking)
        self.ollama_client = OllamaClient(config.ollama)
        
    def convert(self, 
                input_file: Path, 
                output_path: Path,
                custom_prompt: Optional[str] = None) -> bool:
        """Convert any supported file to JSON dataset"""
        
        # Validate file format
        if not FileDetector.is_supported(input_file):
            file_format = FileDetector.detect_format(input_file)
            logger.error(f"Unsupported file format: {file_format.value}")
            logger.info(f"Supported formats: {FileDetector.get_supported_formats()}")
            return False
        
        # Validate Ollama availability
        if not self.ollama_client.is_available():
            logger.error("Ollama server is not available")
            return False
        
        # Process file
        try:
            processed_data = self.universal_processor.process_file(input_file)
            logger.info(f"Successfully parsed {input_file.name} as {processed_data['document_type']}")
        except Exception as e:
            logger.error(f"Failed to process file: {e}")
            return False
        
        # Process chunks
        all_results = []
        chunks = list(self.universal_processor.chunk_document(processed_data))
        
        logger.info(f"Processing {len(chunks)} chunks")
        
        for i, chunk_data in enumerate(tqdm(chunks, desc="Converting chunks")):
            try:
                chunk_df, chunk_metadata = chunk_data
                result = self._process_chunk(chunk_df, chunk_metadata, custom_prompt)
                if result:
                    all_results.extend(result)
            except Exception as e:
                logger.error(f"Error processing chunk {i + 1}: {e}")
                continue
        
        # Save results
        if all_results:
            try:
                self._save_results(all_results, output_path)
                logger.info(f"Successfully converted {len(all_results)} records to {output_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to save results: {e}")
                return False
        else:
            logger.error("No results to save")
            return False
    
    def _process_chunk(self, 
                      chunk_df, 
                      chunk_metadata: Dict[str, Any],
                      custom_prompt: Optional[str]) -> Optional[List[Dict[str, Any]]]:
        """Process a single chunk through the LLM"""
        
        # Get document type and format data appropriately
        document_type = chunk_metadata.get("document_type", "unknown")
        chunk_num = chunk_metadata.get("chunk_number", 1)
        total_chunks = chunk_metadata.get("total_chunks", 1)
        
        # Convert chunk to appropriate string format
        document_data = self.universal_processor.dataframe_to_string(chunk_df, document_type)
        
        # Prepare prompt based on document type
        if custom_prompt:
            user_prompt = custom_prompt.format(
                document_data=document_data,
                document_type=document_type
            )
        else:
            # Select appropriate prompt template based on document type
            if document_type == "tabular":
                prompt_template = self.config.tabular_prompt
            elif document_type == "text":
                prompt_template = self.config.text_prompt
            elif document_type == "presentation":
                prompt_template = self.config.presentation_prompt
            elif document_type in ["pdf", "mixed"]:
                prompt_template = self.config.pdf_prompt
            else:
                # Fallback to generic template
                prompt_template = self.config.user_prompt_template
                user_prompt = prompt_template.format(
                    document_data=document_data,
                    document_type=document_type
                )
            
            # For specific document types, use document_data only
            if document_type in ["tabular", "text", "presentation"] or document_type in ["pdf", "mixed"]:
                user_prompt = prompt_template.format(document_data=document_data)
            else:
                user_prompt = prompt_template.format(
                    document_data=document_data,
                    document_type=document_type
                )
        
        # Add chunk context
        chunk_context = f"\nChunk {chunk_num} of {total_chunks} (rows: {len(chunk_df)})\n"
        user_prompt = chunk_context + user_prompt
        
        try:
            # Call Ollama
            response = self.ollama_client.generate(
                prompt=user_prompt,
                system_prompt=self.config.system_prompt
            )
            
            # Parse LLM response
            llm_output = response.get("response", "").strip()
            return self._parse_llm_response(llm_output, chunk_num)
            
        except Exception as e:
            logger.error(f"Error calling LLM for chunk {chunk_num}: {e}")
            return None
    
    def _parse_llm_response(self, llm_output: str, chunk_num: int) -> Optional[List[Dict[str, Any]]]:
        """Parse and validate LLM response"""
        try:
            # Try to extract JSON from response
            json_start = llm_output.find('[')
            json_end = llm_output.rfind(']') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = llm_output[json_start:json_end]
                parsed_data = json.loads(json_str)
                
                if isinstance(parsed_data, list):
                    logger.info(f"Successfully parsed {len(parsed_data)} records from chunk {chunk_num}")
                    return parsed_data
                else:
                    logger.warning(f"Expected list but got {type(parsed_data)} from chunk {chunk_num}")
                    return [parsed_data] if isinstance(parsed_data, dict) else None
            else:
                # Try parsing the entire response as JSON
                parsed_data = json.loads(llm_output)
                if isinstance(parsed_data, list):
                    return parsed_data
                elif isinstance(parsed_data, dict):
                    return [parsed_data]
                else:
                    logger.error(f"Unexpected JSON structure from chunk {chunk_num}")
                    return None
                    
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from chunk {chunk_num}: {e}")
            logger.debug(f"Raw LLM output: {llm_output[:500]}...")
            return None
    
    def _save_results(self, results: List[Dict[str, Any]], output_path: Path):
        """Save results to file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config.output_format == "jsonl":
            # Save as JSONL (one JSON object per line)
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            # Save as single JSON array
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    
    def validate_setup(self) -> Dict[str, bool]:
        """Validate that all components are properly configured"""
        validation_results = {
            "ollama_available": self.ollama_client.is_available(),
            "model_exists": False
        }
        
        if validation_results["ollama_available"]:
            validation_results["model_exists"] = self.ollama_client.model_exists(
                self.config.ollama.model
            )
        
        return validation_results