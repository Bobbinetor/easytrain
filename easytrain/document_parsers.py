"""Document parsers for different file formats"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json

# Document processing imports
try:
    import openpyxl
    from docx import Document
    import PyPDF2
    from pptx import Presentation
    import fitz  # PyMuPDF
    import tabula
    import camelot
except ImportError as e:
    logging.warning(f"Some document processing libraries not available: {e}")

from .file_detector import FileFormat, FileDetector

logger = logging.getLogger(__name__)


class DocumentParseError(Exception):
    """Custom exception for document parsing errors"""
    pass


class DocumentParser:
    """Base class for document parsers"""
    
    def parse(self, file_path: Path) -> Dict[str, Any]:
        """Parse document and return structured data"""
        raise NotImplementedError
    
    def to_dataframe(self, parsed_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert parsed data to DataFrame for processing"""
        raise NotImplementedError


class CSVParser(DocumentParser):
    """Parser for CSV files"""
    
    def parse(self, file_path: Path) -> Dict[str, Any]:
        """Parse CSV file"""
        try:
            df = pd.read_csv(file_path)
            return {
                "type": "tabular",
                "data": df,
                "metadata": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": list(df.columns)
                }
            }
        except Exception as e:
            raise DocumentParseError(f"Error parsing CSV: {e}")
    
    def to_dataframe(self, parsed_data: Dict[str, Any]) -> pd.DataFrame:
        return parsed_data["data"]


class ExcelParser(DocumentParser):
    """Parser for Excel files (XLSX, XLS)"""
    
    def parse(self, file_path: Path) -> Dict[str, Any]:
        """Parse Excel file"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            sheets_data = {}
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                sheets_data[sheet_name] = df
            
            # If only one sheet, use it directly
            if len(sheets_data) == 1:
                main_df = list(sheets_data.values())[0]
            else:
                # Combine all sheets
                main_df = pd.concat(sheets_data.values(), ignore_index=True)
            
            return {
                "type": "tabular",
                "data": main_df,
                "sheets": sheets_data,
                "metadata": {
                    "sheet_names": excel_file.sheet_names,
                    "total_rows": len(main_df),
                    "total_columns": len(main_df.columns) if not main_df.empty else 0,
                    "column_names": list(main_df.columns) if not main_df.empty else []
                }
            }
        except Exception as e:
            raise DocumentParseError(f"Error parsing Excel: {e}")
    
    def to_dataframe(self, parsed_data: Dict[str, Any]) -> pd.DataFrame:
        return parsed_data["data"]


class PDFParser(DocumentParser):
    """Parser for PDF files"""
    
    def parse(self, file_path: Path) -> Dict[str, Any]:
        """Parse PDF file"""
        try:
            # Try to extract tables first using camelot
            tables_data = self._extract_tables_camelot(file_path)
            
            # If no tables found, try tabula
            if not tables_data:
                tables_data = self._extract_tables_tabula(file_path)
            
            # Extract text content
            text_content = self._extract_text_pymupdf(file_path)
            
            # If we have tables, use them; otherwise convert text to structured format
            if tables_data:
                main_df = pd.concat(tables_data, ignore_index=True)
                data_type = "tabular"
            else:
                # Convert text to simple structure
                lines = [line.strip() for line in text_content.split('\n') if line.strip()]
                main_df = pd.DataFrame({'content': lines})
                data_type = "text"
            
            return {
                "type": data_type,
                "data": main_df,
                "raw_text": text_content,
                "tables_found": len(tables_data) if tables_data else 0,
                "metadata": {
                    "pages": len(text_content.split('\n\n')),  # Rough page count
                    "rows": len(main_df),
                    "columns": len(main_df.columns),
                    "has_tables": bool(tables_data)
                }
            }
        except Exception as e:
            raise DocumentParseError(f"Error parsing PDF: {e}")
    
    def _extract_tables_camelot(self, file_path: Path) -> List[pd.DataFrame]:
        """Extract tables using camelot"""
        try:
            tables = camelot.read_pdf(str(file_path), pages='all')
            return [table.df for table in tables if not table.df.empty]
        except Exception as e:
            logger.debug(f"Camelot table extraction failed: {e}")
            return []
    
    def _extract_tables_tabula(self, file_path: Path) -> List[pd.DataFrame]:
        """Extract tables using tabula"""
        try:
            tables = tabula.read_pdf(str(file_path), pages='all', multiple_tables=True)
            return [df for df in tables if not df.empty]
        except Exception as e:
            logger.debug(f"Tabula table extraction failed: {e}")
            return []
    
    def _extract_text_pymupdf(self, file_path: Path) -> str:
        """Extract text using PyMuPDF"""
        try:
            doc = fitz.open(str(file_path))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.debug(f"PyMuPDF text extraction failed: {e}")
            return ""
    
    def to_dataframe(self, parsed_data: Dict[str, Any]) -> pd.DataFrame:
        return parsed_data["data"]


class WordParser(DocumentParser):
    """Parser for Word documents (DOC, DOCX)"""
    
    def parse(self, file_path: Path) -> Dict[str, Any]:
        """Parse Word document"""
        try:
            doc = Document(file_path)
            
            # Extract paragraphs
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            
            # Extract tables
            tables_data = []
            for table in doc.tables:
                table_data = []
                for row in table.rows:
                    row_data = [cell.text.strip() for cell in row.cells]
                    table_data.append(row_data)
                
                if table_data:
                    # Convert to DataFrame
                    df = pd.DataFrame(table_data[1:], columns=table_data[0])
                    tables_data.append(df)
            
            # Combine data
            if tables_data:
                main_df = pd.concat(tables_data, ignore_index=True)
                data_type = "tabular"
            else:
                # Convert paragraphs to DataFrame
                main_df = pd.DataFrame({'content': paragraphs})
                data_type = "text"
            
            return {
                "type": data_type,
                "data": main_df,
                "paragraphs": paragraphs,
                "tables_found": len(tables_data),
                "metadata": {
                    "paragraph_count": len(paragraphs),
                    "table_count": len(tables_data),
                    "rows": len(main_df),
                    "columns": len(main_df.columns)
                }
            }
        except Exception as e:
            raise DocumentParseError(f"Error parsing Word document: {e}")
    
    def to_dataframe(self, parsed_data: Dict[str, Any]) -> pd.DataFrame:
        return parsed_data["data"]


class PowerPointParser(DocumentParser):
    """Parser for PowerPoint presentations (PPT, PPTX)"""
    
    def parse(self, file_path: Path) -> Dict[str, Any]:
        """Parse PowerPoint presentation"""
        try:
            prs = Presentation(file_path)
            
            slides_content = []
            tables_data = []
            
            for i, slide in enumerate(prs.slides):
                slide_content = {
                    "slide_number": i + 1,
                    "title": "",
                    "content": []
                }
                
                # Extract text from shapes
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        if shape.text_frame and shape.text_frame.paragraphs:
                            # Likely title if it's the first text or short
                            if not slide_content["title"] and len(shape.text.strip()) < 100:
                                slide_content["title"] = shape.text.strip()
                            else:
                                slide_content["content"].append(shape.text.strip())
                    
                    # Extract tables
                    if shape.has_table:
                        table_data = []
                        for row in shape.table.rows:
                            row_data = [cell.text.strip() for cell in row.cells]
                            table_data.append(row_data)
                        
                        if table_data:
                            df = pd.DataFrame(table_data[1:], columns=table_data[0])
                            tables_data.append(df)
                
                slides_content.append(slide_content)
            
            # Create main DataFrame
            if tables_data:
                main_df = pd.concat(tables_data, ignore_index=True)
                data_type = "tabular"
            else:
                # Convert slides to text format
                slide_rows = []
                for slide in slides_content:
                    content_text = " ".join(slide["content"])
                    slide_rows.append({
                        "slide": slide["slide_number"],
                        "title": slide["title"],
                        "content": content_text
                    })
                main_df = pd.DataFrame(slide_rows)
                data_type = "presentation"
            
            return {
                "type": data_type,
                "data": main_df,
                "slides": slides_content,
                "tables_found": len(tables_data),
                "metadata": {
                    "slide_count": len(slides_content),
                    "table_count": len(tables_data),
                    "rows": len(main_df),
                    "columns": len(main_df.columns)
                }
            }
        except Exception as e:
            raise DocumentParseError(f"Error parsing PowerPoint: {e}")
    
    def to_dataframe(self, parsed_data: Dict[str, Any]) -> pd.DataFrame:
        return parsed_data["data"]


class DocumentParserFactory:
    """Factory for creating appropriate document parsers"""
    
    PARSER_MAP = {
        FileFormat.CSV: CSVParser,
        FileFormat.XLSX: ExcelParser,
        FileFormat.XLS: ExcelParser,
        FileFormat.PDF: PDFParser,
        FileFormat.DOC: WordParser,
        FileFormat.DOCX: WordParser,
        FileFormat.PPT: PowerPointParser,
        FileFormat.PPTX: PowerPointParser,
    }
    
    @classmethod
    def create_parser(cls, file_path: Path) -> DocumentParser:
        """Create appropriate parser for file format"""
        file_format = FileDetector.detect_format(file_path)
        
        if file_format not in cls.PARSER_MAP:
            raise DocumentParseError(f"Unsupported file format: {file_format.value}")
        
        parser_class = cls.PARSER_MAP[file_format]
        return parser_class()
    
    @classmethod
    def parse_document(cls, file_path: Path) -> Dict[str, Any]:
        """Parse any supported document format"""
        parser = cls.create_parser(file_path)
        return parser.parse(file_path)
    
    @classmethod
    def document_to_dataframe(cls, file_path: Path) -> pd.DataFrame:
        """Parse document and convert to DataFrame"""
        parser = cls.create_parser(file_path)
        parsed_data = parser.parse(file_path)
        return parser.to_dataframe(parsed_data)