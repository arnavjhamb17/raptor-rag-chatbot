import fitz
import json
import re
import os
from typing import List, Dict, Any, Tuple
from pathlib import Path
import hashlib

class AdvancedPDFParser:
    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        text = re.sub(r'(\.)([A-Z])', r'\1 \2', text)     # Add space after period
        
        return text.strip()
    
    def detect_section_type(self, text: str) -> str:
        """Detect if text is heading, body, citation, etc."""
        text = text.strip()
        
        # Section numbers (e.g., "191.", "192.", "193A.")
        if re.match(r'^\d+[A-Z]*\.\s*[A-Z]', text):
            return 'section_heading'
            
        # Subsection headers (e.g., "(1)", "(2A)", "(iv)")
        if re.match(r'^\([0-9A-Z]+\)', text):
            return 'subsection'
            
        # Main headings (ALL CAPS)
        if re.match(r'^[A-Z][A-Z\s\-—]{10,}\.?$', text):
            return 'main_heading'
            
        # Legal citations and amendments
        if re.search(r'Act No\.|w\.e\.f\.|Sub\. by|Ins\. by|Omtt\. by', text):
            return 'citation'
            
        # Explanations
        if text.startswith('Explanation'):
            return 'explanation'
            
        # Provisos
        if text.startswith('Provided'):
            return 'proviso'
            
        return 'body'
    
    def extract_semantic_chunks(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract semantically meaningful chunks from PDF"""
        doc = fitz.open(pdf_path)
        chunks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            if not text.strip():
                continue
                
            # Split into paragraphs
            paragraphs = re.split(r'\n\s*\n', text.strip())
            
            for para_idx, paragraph in enumerate(paragraphs):
                paragraph = self.clean_text(paragraph)
                
                if len(paragraph) < 20:  # Skip very short paragraphs
                    continue
                
                section_type = self.detect_section_type(paragraph)
                
                # Create chunk ID
                chunk_id = f"page_{page_num+1}_para_{para_idx+1}"
                
                # Split long paragraphs into smaller chunks
                if len(paragraph) > self.chunk_size:
                    sub_chunks = self.split_long_text(paragraph, chunk_id)
                    chunks.extend(sub_chunks)
                else:
                    chunk = self.create_chunk(
                        text=paragraph,
                        chunk_id=chunk_id,
                        page_num=page_num + 1,
                        section_type=section_type,
                        source_file=os.path.basename(pdf_path)
                    )
                    chunks.append(chunk)
        
        doc.close()
        return chunks
    
    def split_long_text(self, text: str, base_id: str) -> List[Dict[str, Any]]:
        """Split long text into overlapping chunks"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunk_id = f"{base_id}_part_{i//self.chunk_size + 1}"
            section_type = self.detect_section_type(chunk_text)
            
            chunk = self.create_chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                page_num=None,  # Will be set by parent
                section_type=section_type,
                source_file=None  # Will be set by parent
            )
            chunks.append(chunk)
            
        return chunks
    
    def create_chunk(self, text: str, chunk_id: str, page_num: int, 
                    section_type: str, source_file: str) -> Dict[str, Any]:
        """Create a standardized chunk object"""
        # Generate content hash for deduplication
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        
        return {
            'id': chunk_id,
            'text': text,
            'content_hash': content_hash,
            'page_number': page_num,
            'section_type': section_type,
            'text_length': len(text),
            'word_count': len(text.split()),
            'metadata': {
                'source_file': source_file,
                'extraction_method': 'semantic_parsing',
                'chunk_type': section_type,
                'is_legal_text': True
            }
        }
    
    def extract_from_case_law(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Specialized extraction for case law PDFs"""
        doc = fitz.open(pdf_path)
        chunks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            if not text.strip():
                continue
            
            # Case law specific patterns
            text = self.clean_text(text)
            
            # Split by legal paragraphs (numbered points, etc.)
            sections = re.split(r'(?=\d+\.|\([a-z]\)|\([0-9]+\))', text)
            
            for i, section in enumerate(sections):
                section = section.strip()
                if len(section) < 30:
                    continue
                
                chunk_id = f"case_{os.path.basename(pdf_path).replace('.pdf', '')}_{page_num+1}_{i+1}"
                section_type = self.detect_case_section_type(section)
                
                chunk = self.create_chunk(
                    text=section,
                    chunk_id=chunk_id,
                    page_num=page_num + 1,
                    section_type=section_type,
                    source_file=os.path.basename(pdf_path)
                )
                chunks.append(chunk)
        
        doc.close()
        return chunks
    
    def detect_case_section_type(self, text: str) -> str:
        """Detect section types specific to case laws"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['held', 'judgment', 'decision']):
            return 'judgment'
        elif any(word in text_lower for word in ['facts', 'background']):
            return 'facts'
        elif 'vs.' in text or 'v.' in text:
            return 'case_title'
        elif any(word in text_lower for word in ['ratio', 'principle']):
            return 'ratio'
        else:
            return 'case_body'
    
    def save_chunks(self, chunks: List[Dict[str, Any]], output_file: str):
        """Save chunks to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(chunks)} chunks to {output_file}")
    
    def save_chunks_jsonl(self, chunks: List[Dict[str, Any]], output_file: str):
        """Save chunks to JSONL file (one JSON object per line)"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(chunks)} chunks to {output_file}")
def main():
    parser = AdvancedPDFParser(chunk_size=800, overlap=100)
    
    # Parse main Income Tax Act PDF
    main_pdf = 'data/Income Tax Act 1961 - TDS Provisions - 2025 Amendment.pdf'
    
    if os.path.exists(main_pdf):
        print("Parsing main Income Tax Act PDF...")
        chunks = parser.extract_semantic_chunks(main_pdf)
        parser.save_chunks(chunks, 'income_tax_chunks.json')
        parser.save_chunks_jsonl(chunks, 'income_tax_chunks.jsonl')
        
        # Print statistics
        section_types = {}
        for chunk in chunks:
            section_type = chunk['section_type']
            section_types[section_type] = section_types.get(section_type, 0) + 1
        
        print(f"\nExtracted {len(chunks)} total chunks")
        print("Section type distribution:")
        for stype, count in sorted(section_types.items()):
            print(f"  {stype}: {count}")
    
    # Parse case law PDFs
    case_law_dir = 'data/Case Laws_194C, 194J, 194I, 194H'
    all_case_chunks = []
    
    if os.path.exists(case_law_dir):
        print(f"\nParsing case law PDFs from {case_law_dir}...")
        
        for root, dirs, files in os.walk(case_law_dir):
            for file in files:
                if file.endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    print(f"Processing: {file}")
                    
                    try:
                        case_chunks = parser.extract_from_case_law(pdf_path)
                        all_case_chunks.extend(case_chunks)
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
        
        if all_case_chunks:
            parser.save_chunks(all_case_chunks, 'case_law_chunks.json')
            parser.save_chunks_jsonl(all_case_chunks, 'case_law_chunks.jsonl')
            
            print(f"\nExtracted {len(all_case_chunks)} chunks from case law PDFs")

if __name__ == "__main__":
    main()

