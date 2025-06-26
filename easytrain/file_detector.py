"""File format detection and routing"""

import mimetypes
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FileFormat(Enum):
    """Supported file formats"""
    CSV = "csv"
    XLSX = "xlsx"
    XLS = "xls"
    DOC = "doc"
    DOCX = "docx"
    PDF = "pdf"
    PPT = "ppt"
    PPTX = "pptx"
    TXT = "txt"
    UNKNOWN = "unknown"


class FileDetector:
    """Detects file format and provides metadata"""
    
    # MIME type mappings
    MIME_MAPPINGS = {
        'text/csv': FileFormat.CSV,
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': FileFormat.XLSX,
        'application/vnd.ms-excel': FileFormat.XLS,
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': FileFormat.DOCX,
        'application/msword': FileFormat.DOC,
        'application/pdf': FileFormat.PDF,
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': FileFormat.PPTX,
        'application/vnd.ms-powerpoint': FileFormat.PPT,
        'text/plain': FileFormat.TXT
    }
    
    # Extension fallback mappings
    EXTENSION_MAPPINGS = {
        '.csv': FileFormat.CSV,
        '.xlsx': FileFormat.XLSX,
        '.xls': FileFormat.XLS,
        '.docx': FileFormat.DOCX,
        '.doc': FileFormat.DOC,
        '.pdf': FileFormat.PDF,
        '.pptx': FileFormat.PPTX,
        '.ppt': FileFormat.PPT,
        '.txt': FileFormat.TXT
    }
    
    @classmethod
    def detect_format(cls, file_path: Path) -> FileFormat:
        """Detect file format using MIME type and extension"""
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return FileFormat.UNKNOWN
        
        # Try MIME type detection first
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type and mime_type in cls.MIME_MAPPINGS:
            detected_format = cls.MIME_MAPPINGS[mime_type]
            logger.info(f"Detected format via MIME type: {detected_format.value}")
            return detected_format
        
        # Fallback to extension detection
        extension = file_path.suffix.lower()
        if extension in cls.EXTENSION_MAPPINGS:
            detected_format = cls.EXTENSION_MAPPINGS[extension]
            logger.info(f"Detected format via extension: {detected_format.value}")
            return detected_format
        
        logger.warning(f"Unknown file format for: {file_path}")
        return FileFormat.UNKNOWN
    
    @classmethod
    def get_file_info(cls, file_path: Path) -> Dict[str, Any]:
        """Get comprehensive file information"""
        
        if not file_path.exists():
            return {"error": "File not found"}
        
        stat = file_path.stat()
        
        return {
            "path": str(file_path),
            "name": file_path.name,
            "extension": file_path.suffix.lower(),
            "format": cls.detect_format(file_path),
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": stat.st_mtime
        }
    
    @classmethod
    def is_supported(cls, file_path: Path) -> bool:
        """Check if file format is supported"""
        return cls.detect_format(file_path) != FileFormat.UNKNOWN
    
    @classmethod
    def get_supported_formats(cls) -> list:
        """Get list of supported file formats"""
        return [fmt.value for fmt in FileFormat if fmt != FileFormat.UNKNOWN]