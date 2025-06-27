#!/usr/bin/env python3
"""Analyze the current workflow bottlenecks"""

import sys
import time
sys.path.append('/home/fitnesslab/easytrain')

from easytrain.document_parsers import PDFParser
from easytrain.universal_processor import UniversalProcessor
from easytrain.config import ChunkingConfig
from pathlib import Path

def analyze_workflow():
    pdf_path = Path("/home/fitnesslab/easytrain/resources/lm35-3-1-5.pdf")
    parser = PDFParser()
    
    print("🔍 WORKFLOW ANALYSIS")
    print("=" * 50)
    
    # Step 1: PDF Parsing
    print("\n📄 Step 1: PDF Parsing")
    start_time = time.time()
    
    try:
        parsed_data = parser.parse(pdf_path)
        parse_time = time.time() - start_time
        
        print(f"✅ Parsing completed in {parse_time:.2f} seconds")
        print(f"   Document type: {parsed_data['type']}")
        print(f"   Raw text length: {len(parsed_data['raw_text'])} characters")
        print(f"   Structured rows: {len(parsed_data['data'])}")
        print(f"   Used OCR: {parsed_data['metadata'].get('used_ocr', False)}")
        
        # Step 2: Chunking Analysis
        print("\n🔪 Step 2: Chunking Analysis")
        
        # Test different chunk sizes
        chunk_configs = [
            {"size": 5, "text_len": 1500},
            {"size": 8, "text_len": 2500}, 
            {"size": 10, "text_len": 3000},
            {"size": 15, "text_len": 4000}
        ]
        
        for config in chunk_configs:
            chunking_config = ChunkingConfig(
                max_rows_per_chunk=config["size"],
                max_text_length=config["text_len"]
            )
            
            processor = UniversalProcessor(chunking_config)
            
            processed_data = {
                "file_info": {"path": str(pdf_path), "name": pdf_path.name},
                "parsed_data": parsed_data,
                "document_type": parsed_data["type"]
            }
            
            start_chunk_time = time.time()
            chunks = list(processor.chunk_document(processed_data))
            chunk_time = time.time() - start_chunk_time
            
            # Analyze chunk content
            total_chars = 0
            for chunk_df, metadata in chunks:
                chunk_text = processor.dataframe_to_string(chunk_df, parsed_data["type"])
                total_chars += len(chunk_text)
            
            avg_chars_per_chunk = total_chars / len(chunks) if chunks else 0
            
            print(f"   Config {config['size']} rows, {config['text_len']} max chars:")
            print(f"   - Chunks created: {len(chunks)}")
            print(f"   - Chunking time: {chunk_time:.3f}s")
            print(f"   - Avg chars/chunk: {avg_chars_per_chunk:.0f}")
            print(f"   - Estimated LLM time: {len(chunks) * 20:.0f}s ({len(chunks)} chunks × 20s)")
            print()
        
        # Step 3: Content Quality Analysis
        print("\n📊 Step 3: Content Quality Analysis")
        
        if parsed_data['type'] == 'text':
            lines = parsed_data['raw_text'].split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            
            # Analyze line types
            short_lines = [line for line in non_empty_lines if len(line.strip()) < 10]
            medium_lines = [line for line in non_empty_lines if 10 <= len(line.strip()) < 50]
            long_lines = [line for line in non_empty_lines if len(line.strip()) >= 50]
            
            print(f"   Total lines: {len(lines)}")
            print(f"   Non-empty lines: {len(non_empty_lines)}")
            print(f"   Short lines (<10 chars): {len(short_lines)} ({len(short_lines)/len(non_empty_lines)*100:.1f}%)")
            print(f"   Medium lines (10-50 chars): {len(medium_lines)} ({len(medium_lines)/len(non_empty_lines)*100:.1f}%)")
            print(f"   Long lines (>50 chars): {len(long_lines)} ({len(long_lines)/len(non_empty_lines)*100:.1f}%)")
            
            # Show sample of different line types
            print(f"\n   Sample short lines: {short_lines[:3]}")
            print(f"   Sample medium lines: {medium_lines[:3]}")
            print(f"   Sample long lines: {[line[:100]+'...' for line in long_lines[:2]]}")
        
        print("\n💡 RECOMMENDATIONS:")
        print("   1. Use smaller chunks (5-8 rows) for faster processing")
        print("   2. Filter out very short lines (likely OCR noise)")
        print("   3. Consider parallel processing of chunks")
        print("   4. Pre-process OCR text to remove noise")
        
    except Exception as e:
        print(f"💥 Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_workflow()