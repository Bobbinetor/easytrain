#!/usr/bin/env python3
"""Debug the chunking process for OCR documents"""

import sys
sys.path.append('/home/fitnesslab/easytrain')

from easytrain.universal_processor import UniversalProcessor
from easytrain.config import ChunkingConfig
from easytrain.document_parsers import PDFParser
from pathlib import Path

def debug_chunking():
    pdf_path = Path("/home/fitnesslab/easytrain/resources/lm35-3-1-5.pdf")
    
    print("🔍 CHUNKING DEBUG ANALYSIS")
    print("=" * 60)
    
    # Parse document
    parser = PDFParser()
    parsed_data = parser.parse(pdf_path)
    
    print(f"📄 Document type: {parsed_data['type']}")
    print(f"📊 Data shape: {parsed_data['data'].shape}")
    print(f"📝 Raw text length: {len(parsed_data['raw_text'])} characters")
    print(f"🔍 Used OCR: {parsed_data['metadata'].get('used_ocr', False)}")
    
    # Setup processor with different configs
    configs = [
        {"name": "Small", "chunk_size": 1500, "overlap": 150},
        {"name": "Medium", "chunk_size": 2000, "overlap": 200},
        {"name": "Large", "chunk_size": 3000, "overlap": 300}
    ]
    
    for config in configs:
        print(f"\n📦 Testing {config['name']} chunks:")
        print(f"   Chunk size: {config['chunk_size']} chars, Overlap: {config['overlap']}")
        
        chunking_config = ChunkingConfig(
            chunk_size=config['chunk_size'],
            overlap_size=config['overlap']
        )
        
        processor = UniversalProcessor(chunking_config)
        
        # Create processed data structure
        processed_data = {
            "file_info": {"path": str(pdf_path), "name": pdf_path.name, "used_ocr": parsed_data['metadata'].get('used_ocr', False)},
            "parsed_data": parsed_data,
            "document_type": parsed_data["type"]
        }
        
        # Generate chunks
        chunks = list(processor.chunk_document(processed_data))
        
        print(f"   📊 Number of chunks: {len(chunks)}")
        
        if chunks:
            # Analyze first few chunks
            total_chars = sum(len(chunk_text) for chunk_text, _ in chunks)
            avg_chars = total_chars / len(chunks)
            
            print(f"   📏 Average chunk size: {avg_chars:.0f} characters")
            print(f"   📋 Total processed chars: {total_chars}")
            
            # Show first chunk sample
            first_chunk_text, first_metadata = chunks[0]
            print(f"   🔍 First chunk preview (first 200 chars):")
            print(f"   '{first_chunk_text[:200]}...'")
            
            # Show chunk distribution
            chunk_sizes = [len(chunk_text) for chunk_text, _ in chunks]
            min_size, max_size = min(chunk_sizes), max(chunk_sizes)
            print(f"   📊 Chunk size range: {min_size} - {max_size} characters")
            
            # Estimate processing time
            estimated_time = len(chunks) * 20  # 20 seconds per chunk
            print(f"   ⏱️  Estimated processing time: {estimated_time} seconds ({estimated_time/60:.1f} minutes)")
        
        print("-" * 40)
    
    # Analyze the actual text conversion
    print(f"\n🔤 TEXT CONVERSION ANALYSIS:")
    full_text = processor._document_to_text(parsed_data, parsed_data['type'])
    print(f"   Converted text length: {len(full_text)} characters")
    print(f"   Lines in converted text: {len(full_text.split(chr(10)))}")
    
    # Show sample of converted text
    print(f"   📝 First 300 chars of converted text:")
    print(f"   '{full_text[:300]}...'")

if __name__ == "__main__":
    debug_chunking()