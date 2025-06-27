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
        
        print(f"🔄 Processing {len(chunks)} chunks from {processed_data['document_type']} document")
        logger.info(f"Processing {len(chunks)} chunks")
        
        for i, chunk_data in enumerate(tqdm(chunks, desc="Converting chunks")):
            try:
                chunk_text, chunk_metadata = chunk_data
                chunk_size = chunk_metadata.get("chunk_size", len(chunk_text))
                print(f"📄 Processing chunk {i+1}/{len(chunks)} - {chunk_size} characters")
                result = self._process_chunk_text(chunk_text, chunk_metadata, custom_prompt)
                if result:
                    print(f"✅ Chunk {i+1} generated {len(result)} records")
                    all_results.extend(result)
                else:
                    print(f"❌ Chunk {i+1} failed to generate records")
            except Exception as e:
                print(f"💥 Error processing chunk {i + 1}: {e}")
                logger.error(f"Error processing chunk {i + 1}: {e}")
                continue
        
        # Save results
        if all_results:
            try:
                print(f"💾 Saving {len(all_results)} total records...")
                self._save_results(all_results, output_path)
                print(f"✅ Successfully saved to {output_path}")
                logger.info(f"Successfully converted {len(all_results)} records to {output_path}")
                return True
            except Exception as e:
                print(f"💥 Failed to save results: {e}")
                logger.error(f"Failed to save results: {e}")
                return False
        else:
            print("❌ No results to save")
            logger.error("No results to save")
            return False
    
    def _process_chunk_text(self, 
                           chunk_text: str, 
                           chunk_metadata: Dict[str, Any],
                           custom_prompt: Optional[str]) -> Optional[List[Dict[str, Any]]]:
        """Process a single text chunk through the LLM"""
        
        # Get document type and metadata
        document_type = chunk_metadata.get("document_type", "unknown")
        chunk_num = chunk_metadata.get("chunk_number", 1)
        total_chunks = chunk_metadata.get("total_chunks", 1)
        chunk_size = chunk_metadata.get("chunk_size", len(chunk_text))
        
        # Prepare prompt based on document type
        if custom_prompt:
            user_prompt = custom_prompt.format(
                document_data=chunk_text,
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
                    document_data=chunk_text,
                    document_type=document_type
                )
            
            # For specific document types, use document_data only
            if document_type in ["tabular", "text", "presentation"] or document_type in ["pdf", "mixed"]:
                user_prompt = prompt_template.format(document_data=chunk_text)
            else:
                user_prompt = prompt_template.format(
                    document_data=chunk_text,
                    document_type=document_type
                )
        
        # Add chunk context
        chunk_context = f"\nChunk {chunk_num} of {total_chunks} ({chunk_size} characters)\n"
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
        """Parse and validate LLM response with enhanced error handling"""
        try:
            # First, try to clean and extract JSON from response
            json_data = self._extract_json_from_text(llm_output)
            
            if json_data:
                if isinstance(json_data, list):
                    logger.info(f"Successfully parsed {len(json_data)} records from chunk {chunk_num}")
                    return json_data
                elif isinstance(json_data, dict):
                    logger.info(f"Successfully parsed 1 record from chunk {chunk_num}")
                    return [json_data]
                else:
                    logger.warning(f"Expected list or dict but got {type(json_data)} from chunk {chunk_num}")
                    return None
            else:
                logger.error(f"No valid JSON found in chunk {chunk_num}")
                logger.debug(f"Raw LLM output: {llm_output[:500]}...")
                return None
                    
        except Exception as e:
            logger.error(f"Unexpected error parsing chunk {chunk_num}: {e}")
            logger.debug(f"Raw LLM output: {llm_output[:500]}...")
            return None
    
    def _extract_json_from_text(self, text: str) -> Optional[Any]:
        """Enhanced JSON extraction with multiple fallback strategies"""
        import re
        
        # Strategy 1: Try to find JSON array
        json_start = text.find('[')
        json_end = text.rfind(']') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = text[json_start:json_end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Strategy 2: Try to find JSON object
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = text[json_start:json_end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Try parsing the entire response
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        # Strategy 4: Find and parse multiple JSON objects (JSONL-style)
        json_objects = []
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    obj = json.loads(line)
                    json_objects.append(obj)
                except json.JSONDecodeError:
                    continue
        
        if json_objects:
            return json_objects
        
        # Strategy 5: Use regex to find JSON-like structures
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text)
        
        for match in matches:
            try:
                obj = json.loads(match)
                if isinstance(obj, dict) and any(key in obj for key in ['instruction', 'input', 'output']):
                    json_objects.append(obj)
            except json.JSONDecodeError:
                continue
        
        if json_objects:
            return json_objects
        
        # Strategy 6: Clean common formatting issues and try again
        cleaned_text = self._clean_json_text(text)
        if cleaned_text != text:
            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _clean_json_text(self, text: str) -> str:
        """Clean common JSON formatting issues"""
        import re
        
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Remove common prefixes
        text = re.sub(r'^Here is the.*?:\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'^The JSON.*?:\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Fix common quote issues
        text = re.sub(r'"([^"]+)"(\s*:\s*)"([^"]*)"', r'"\1": "\3"', text)
        
        # Remove trailing commas
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        # Fix double quotes in values
        text = re.sub(r':\s*"([^"]*)"([^",}\]]*)"([^",}\]]*)"', r': "\1\2\3"', text)
        
        return text.strip()
    
    def _validate_schema(self, results: List[Dict[str, Any]]) -> bool:
        """Validate that all results conform to instruction-input-output schema"""
        required_fields = {"instruction", "input", "output"}
        
        for i, item in enumerate(results):
            if not isinstance(item, dict):
                logger.error(f"Item {i} is not a dictionary")
                return False
            
            item_fields = set(item.keys())
            if not required_fields.issubset(item_fields):
                missing = required_fields - item_fields
                logger.error(f"Item {i} missing required fields: {missing}")
                return False
                
            # Check that all required fields have string values
            for field in required_fields:
                if not isinstance(item[field], str):
                    logger.error(f"Item {i} field '{field}' is not a string: {type(item[field])}")
                    return False
        
        logger.info(f"Schema validation passed for {len(results)} records")
        return True
    
    def _save_results(self, results: List[Dict[str, Any]], output_path: Path):
        """Save results to file with schema validation"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate schema
        print(f"🔍 Validating schema for {len(results)} records...")
        schema_valid = self._validate_schema(results)
        
        if schema_valid and self.config.output_format == "json":
            # Save as single JSON array (preferred format)
            print("✅ Schema validation passed, saving as JSON array")
            logger.info("Schema validation passed, saving as JSON array")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        else:
            # Fallback to JSONL format
            if not schema_valid:
                print("⚠️ Schema validation failed, falling back to JSONL format")
                logger.warning("Schema validation failed, falling back to JSONL format")
            else:
                print("📝 Using JSONL format as requested")
                logger.info("Using JSONL format as requested")
            
            # Change extension to .jsonl if needed
            if output_path.suffix == '.json':
                output_path = output_path.with_suffix('.jsonl')
                print(f"📝 Changed extension to .jsonl: {output_path}")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
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