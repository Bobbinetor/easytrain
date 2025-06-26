"""Universal document processor that handles all supported formats"""

import pandas as pd
from typing import List, Iterator, Optional, Dict, Any, Tuple
from pathlib import Path
import logging

from .config import ChunkingConfig
from .file_detector import FileDetector, FileFormat
from .document_parsers import DocumentParserFactory

logger = logging.getLogger(__name__)


class UniversalProcessor:
    """Handles any supported file format with chunking capabilities"""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
    
    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process any supported file format"""
        if not FileDetector.is_supported(file_path):
            raise ValueError(f"Unsupported file format: {file_path}")
        
        try:
            # Parse the document
            parsed_data = DocumentParserFactory.parse_document(file_path)
            
            # Get file info
            file_info = FileDetector.get_file_info(file_path)
            
            # Combine information
            result = {
                "file_info": file_info,
                "parsed_data": parsed_data,
                "document_type": parsed_data.get("type", "unknown")
            }
            
            logger.info(f"Successfully processed {file_path.name} as {result['document_type']}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise
    
    def chunk_document(self, processed_data: Dict[str, Any]) -> Iterator[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Chunk processed document data"""
        document_type = processed_data["document_type"]
        parsed_data = processed_data["parsed_data"]
        file_info = processed_data["file_info"]
        
        if document_type == "tabular":
            # Handle tabular data (CSV, Excel tables, PDF tables, etc.)
            yield from self._chunk_tabular_data(parsed_data["data"], file_info)
        
        elif document_type in ["text", "presentation"]:
            # Handle text-based documents
            yield from self._chunk_text_data(parsed_data["data"], file_info, document_type)
        
        else:
            # Fallback: treat as single chunk
            yield parsed_data["data"], {
                "chunk_number": 1,
                "total_chunks": 1,
                "document_type": document_type,
                "file_info": file_info
            }
    
    def _chunk_tabular_data(self, df: pd.DataFrame, file_info: Dict[str, Any]) -> Iterator[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Chunk tabular data (similar to original CSV chunking)"""
        total_rows = len(df)
        chunk_size = self.config.max_rows_per_chunk
        overlap = self.config.overlap_rows
        
        if total_rows <= chunk_size:
            yield df, {
                "chunk_number": 1,
                "total_chunks": 1,
                "document_type": "tabular",
                "file_info": file_info,
                "rows": total_rows
            }
            return
        
        start_idx = 0
        chunk_num = 0
        total_chunks = (total_rows + chunk_size - 1) // chunk_size  # Ceiling division
        
        while start_idx < total_rows:
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk = df.iloc[start_idx:end_idx].copy()
            
            chunk_num += 1
            
            yield chunk, {
                "chunk_number": chunk_num,
                "total_chunks": total_chunks,
                "document_type": "tabular",
                "file_info": file_info,
                "rows": len(chunk),
                "start_row": start_idx,
                "end_row": end_idx - 1
            }
            
            start_idx = end_idx - overlap
            if start_idx <= 0:
                start_idx = end_idx
    
    def _chunk_text_data(self, df: pd.DataFrame, file_info: Dict[str, Any], doc_type: str) -> Iterator[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Chunk text-based data"""
        total_rows = len(df)
        
        # For text documents, we might want to chunk by content length as well
        if self.config.preserve_structure:
            # Try to keep logical units together (paragraphs, slides, etc.)
            yield from self._chunk_by_structure(df, file_info, doc_type)
        else:
            # Simple row-based chunking
            yield from self._chunk_tabular_data(df, file_info)
    
    def _chunk_by_structure(self, df: pd.DataFrame, file_info: Dict[str, Any], doc_type: str) -> Iterator[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Chunk while preserving document structure"""
        total_rows = len(df)
        max_rows = self.config.max_rows_per_chunk
        max_text_length = self.config.max_text_length
        
        current_chunk = []
        current_text_length = 0
        chunk_num = 0
        
        for idx, row in df.iterrows():
            # Estimate text length for this row
            row_text_length = sum(len(str(val)) for val in row.values)
            
            # Check if adding this row would exceed limits
            if (len(current_chunk) >= max_rows or 
                current_text_length + row_text_length > max_text_length) and current_chunk:
                
                # Yield current chunk
                chunk_num += 1
                chunk_df = pd.DataFrame(current_chunk)
                
                yield chunk_df, {
                    "chunk_number": chunk_num,
                    "total_chunks": -1,  # Will be updated later
                    "document_type": doc_type,
                    "file_info": file_info,
                    "rows": len(chunk_df),
                    "text_length": current_text_length
                }
                
                # Reset for next chunk
                current_chunk = []
                current_text_length = 0
            
            # Add current row to chunk
            current_chunk.append(row.to_dict())
            current_text_length += row_text_length
        
        # Handle remaining data
        if current_chunk:
            chunk_num += 1
            chunk_df = pd.DataFrame(current_chunk)
            
            yield chunk_df, {
                "chunk_number": chunk_num,
                "total_chunks": chunk_num,  # Now we know the total
                "document_type": doc_type,
                "file_info": file_info,
                "rows": len(chunk_df),
                "text_length": current_text_length
            }
    
    def dataframe_to_string(self, df: pd.DataFrame, document_type: str) -> str:
        """Convert dataframe to string appropriate for the document type"""
        if document_type == "tabular":
            # For tabular data, use CSV format
            return df.to_csv(index=False)
        
        elif document_type == "text":
            # For text documents, format as readable text
            if 'content' in df.columns:
                return '\n'.join(df['content'].astype(str))
            else:
                # Fallback to all columns
                return '\n'.join(df.apply(lambda row: ' '.join(row.astype(str)), axis=1))
        
        elif document_type == "presentation":
            # For presentations, format with slide structure
            formatted_lines = []
            for _, row in df.iterrows():
                if 'title' in row and 'content' in row:
                    formatted_lines.append(f"Slide {row.get('slide', '?')}: {row['title']}")
                    formatted_lines.append(f"Content: {row['content']}")
                    formatted_lines.append("")  # Empty line between slides
                else:
                    formatted_lines.append(' '.join(str(val) for val in row.values))
            return '\n'.join(formatted_lines)
        
        else:
            # Generic fallback
            return df.to_string(index=False)
    
    def get_chunk_info(self, chunk_data: Tuple[pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
        """Get information about a chunk"""
        df, metadata = chunk_data
        
        base_info = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "memory_usage": df.memory_usage(deep=True).sum()
        }
        
        # Merge with chunk metadata
        base_info.update(metadata)
        return base_info