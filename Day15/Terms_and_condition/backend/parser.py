import re
from typing import List, Dict, Any
from pypdf import PdfReader

class SmartPDFParser:
    """
    Parses PDF documents and splits them into logical clauses or sections, 
    preserving the structure of legal terms & conditions.
    """
    
    # Common legal section headers (e.g., "Section 1", "Clause 4.2", "12. Indemnification")
    SECTION_PATTERNS = [
        r'(?i)^\s*(?:section|clause|article|para|paragraph)\s+(\d+(?:\.\d+)*)\b',
        r'^\s*(\d+(?:\.\d+)+)\s+([A-Z][A-Za-z0-9\s\-,&\(\)/]+)',
        r'^\s*(\d+)\.\s+([A-Z][A-Za-z0-9\s\-,&\(\)/]{3,})'
    ]

    @staticmethod
    def extract_text_by_page(pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extracts raw text page-by-page from a PDF file.
        """
        reader = PdfReader(pdf_path)
        pages_content = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages_content.append({
                    "page_num": i + 1,
                    "text": text
                })
        return pages_content

    def chunk_document(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Reads a PDF and segments it into semantic legal chunks (clauses/sections).
        """
        pages = self.extract_text_by_page(pdf_path)
        if not pages:
            return []

        # Concatenate text with page markers to keep track of page numbers
        full_text = ""
        page_offsets = []
        current_len = 0
        
        for p in pages:
            page_text = p["text"]
            full_text += page_text + "\n\n"
            page_offsets.append({
                "start_idx": current_len,
                "end_idx": current_len + len(page_text) + 2,
                "page_num": p["page_num"]
            })
            current_len += len(page_text) + 2

        # Helper to find page number for a character index
        def get_page_number(char_idx: int) -> int:
            for offset in page_offsets:
                if offset["start_idx"] <= char_idx < offset["end_idx"]:
                    return offset["page_num"]
            return pages[-1]["page_num"] if pages else 1

        # Locate section breaks using regex
        lines = full_text.split('\n')
        chunks = []
        current_chunk_title = "Introduction"
        current_chunk_lines = []
        current_chunk_start_idx = 0
        current_chunk_page = 1

        # Check if line looks like a header
        def is_header(line: str) -> bool:
            # Avoid long sentences being flagged as headers
            if len(line.strip()) > 80:
                return False
            for pattern in self.SECTION_PATTERNS:
                if re.match(pattern, line.strip()):
                    return True
            return False

        line_char_counter = 0
        for line in lines:
            if is_header(line):
                # If we have accumulated content, save it as a chunk
                if current_chunk_lines:
                    chunk_text = "\n".join(current_chunk_lines).strip()
                    if chunk_text:
                        chunks.append({
                            "title": current_chunk_title,
                            "text": chunk_text,
                            "page_num": current_chunk_page
                        })
                
                # Start new chunk
                current_chunk_title = line.strip()
                current_chunk_lines = [line]
                current_chunk_start_idx = line_char_counter
                current_chunk_page = get_page_number(line_char_counter)
            else:
                current_chunk_lines.append(line)
            
            line_char_counter += len(line) + 1 # +1 for newline character

        # Append final chunk
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines).strip()
            if chunk_text:
                chunks.append({
                    "title": current_chunk_title,
                    "text": chunk_text,
                    "page_num": current_chunk_page
                })

        # Fallback: If no structured sections were found (e.g. less than 3 chunks),
        # split by paragraph clusters to maintain logical chunking.
        if len(chunks) < 3:
            chunks = []
            paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]
            
            temp_chunk = []
            temp_word_count = 0
            chunk_idx = 1
            chunk_page = 1
            
            for i, para in enumerate(paragraphs):
                # Approximate page based on paragraph index
                approx_char_idx = full_text.find(para[:50]) if len(para) > 50 else 0
                para_page = get_page_number(approx_char_idx) if approx_char_idx > 0 else 1
                
                temp_chunk.append(para)
                temp_word_count += len(para.split())
                
                # Split when chunk is around 400-600 words, or page changes
                if temp_word_count >= 500 or (i == len(paragraphs) - 1):
                    chunks.append({
                        "title": f"Clause Group {chunk_idx}",
                        "text": "\n\n".join(temp_chunk),
                        "page_num": chunk_page
                    })
                    temp_chunk = []
                    temp_word_count = 0
                    chunk_idx += 1
                    chunk_page = para_page

        # Standardize ids and remove noise
        formatted_chunks = []
        for idx, chunk in enumerate(chunks):
            # Clean title
            clean_title = re.sub(r'[\s\.\-\:]+', '_', chunk["title"].strip()).strip('_')
            clean_title = re.sub(r'[^a-zA-Z0-9_]', '', clean_title)[:50]
            if not clean_title:
                clean_title = f"section_{idx+1}"
            
            formatted_chunks.append({
                "id": f"{clean_title.lower()}_{idx+1}",
                "title": chunk["title"].strip(),
                "text": chunk["text"],
                "page_num": chunk["page_num"]
            })

        return formatted_chunks

if __name__ == "__main__":
    # Test stub
    import sys
    if len(sys.argv) > 1:
        parser = SmartPDFParser()
        result = parser.chunk_document(sys.argv[1])
        print(f"Extracted {len(result)} chunks:")
        for r in result[:3]:
            print(f"- ID: {r['id']} | Title: {r['title']} | Page: {r['page_num']}")
            print(f"  Snippet: {r['text'][:150]}...")
