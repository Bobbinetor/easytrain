"""CSV processing and chunking functionality"""

import pandas as pd
from typing import List, Iterator, Optional
from pathlib import Path
import logging

from .config import ChunkingConfig

logger = logging.getLogger(__name__)


class CSVProcessor:
    """Handles CSV file reading and chunking operations"""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
    
    def read_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Read CSV file with error handling"""
        try:
            df = pd.read_csv(file_path, **kwargs)
            logger.info(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            raise
    
    def chunk_dataframe(self, df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """Split dataframe into chunks with optional overlap"""
        total_rows = len(df)
        chunk_size = self.config.max_rows_per_chunk
        overlap = self.config.overlap_rows
        
        if total_rows <= chunk_size:
            yield df
            return
        
        start_idx = 0
        chunk_num = 0
        
        while start_idx < total_rows:
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk = df.iloc[start_idx:end_idx].copy()
            
            logger.info(f"Creating chunk {chunk_num + 1}: rows {start_idx}-{end_idx-1}")
            yield chunk
            
            chunk_num += 1
            start_idx = end_idx - overlap
            
            # Prevent infinite loop if overlap is too large
            if start_idx <= 0:
                start_idx = end_idx
    
    def dataframe_to_csv_string(self, df: pd.DataFrame, include_index: bool = False) -> str:
        """Convert dataframe chunk to CSV string for LLM processing"""
        return df.to_csv(index=include_index)
    
    def get_chunk_info(self, df: pd.DataFrame) -> dict:
        """Get information about the dataframe chunk"""
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "memory_usage": df.memory_usage(deep=True).sum()
        }