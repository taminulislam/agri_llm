"""
Semantic text chunking for agricultural dataset.

Creates coherent topic-based chunks from page-level JSONL data.
"""

import re
import json
from typing import List, Dict
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict


@dataclass
class Chunk:
    """Semantic chunk with metadata"""
    id: str
    text: str
    doc: str
    page_range: List[int]
    topic: str
    char_count: int
    sentence_count: int
    keywords: List[str]
    source_pages: List[str]


class SemanticChunker:
    """
    Create semantic chunks from page-level agricultural data.

    Strategy:
    1. Combine pages from same document
    2. Split by section headers (detect with regex)
    3. Further split long sections by paragraph boundaries
    4. Ensure chunks are 1500-3000 characters (optimal for context)
    5. Add 200-char overlap for continuity
    """

    def __init__(self,
                 target_chunk_size: int = 2000,
                 min_chunk_size: int = 1000,
                 max_chunk_size: int = 3500,
                 overlap_size: int = 200):
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size

    def detect_section_headers(self, text: str) -> List[int]:
        """
        Detect section boundaries using patterns common in agricultural texts.

        Patterns:
        - All caps headers: "CORN PRODUCTION"
        - Numbered sections: "2.1 Hybrid Selection"
        - Title case on own line: "Weed Management Strategies"
        """
        patterns = [
            r'\n([A-Z][A-Z\s]{10,})\n',  # ALL CAPS headers
            r'\n(\d+\.?\d*\s+[A-Z][^\n]{5,50})\n',  # Numbered sections
            r'\n([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})\n(?=[A-Z\(])',  # Title case
        ]

        boundaries = [0]
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                boundaries.append(match.start())

        boundaries.append(len(text))
        return sorted(set(boundaries))

    def split_by_paragraphs(self, text: str, max_size: int) -> List[str]:
        """Split text by paragraph boundaries when too long"""
        if len(text) <= max_size:
            return [text]

        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) < max_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def extract_topic(self, text: str, doc_name: str) -> str:
        """Extract likely topic from first few lines or doc name"""
        # Try to find title in first 200 chars
        first_lines = text[:200].split('\n')
        for line in first_lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 80 and line[0].isupper():
                return line

        # Fall back to doc name
        topic = doc_name.replace('_', ' ').replace('-', ' ')
        topic = re.sub(r'\d{4}web', '', topic)
        topic = re.sub(r'chapter\d+', 'Agricultural Chapter', topic)
        return topic.title()

    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract key agricultural terms using frequency and domain patterns"""
        # Common agricultural terms to prioritize
        ag_terms = {
            'corn', 'soybean', 'wheat', 'nitrogen', 'phosphorus', 'potassium',
            'herbicide', 'pesticide', 'insect', 'disease', 'weed', 'crop',
            'soil', 'water', 'yield', 'plant', 'seed', 'hybrid', 'variety',
            'fertilizer', 'management', 'control', 'pest', 'tillage', 'planting',
            'harvest', 'acre', 'field', 'farmer', 'growing', 'season'
        }

        # Extract words
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())

        # Count frequency
        word_freq = {}
        for word in words:
            if word in ag_terms:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:top_k]]

    def count_sentences(self, text: str) -> int:
        """Simple sentence counter"""
        # Count periods, exclamation marks, and question marks
        sentence_enders = text.count('.') + text.count('!') + text.count('?')
        return max(1, sentence_enders)

    def chunk_document(self, pages: List[Dict]) -> List[Chunk]:
        """
        Create semantic chunks from document pages.

        Args:
            pages: List of page records from JSONL

        Returns:
            List of semantic chunks with metadata
        """
        # Combine pages into single text
        full_text = "\n\n".join(page['text'] for page in pages)
        doc_name = pages[0]['doc']

        # Detect section boundaries
        boundaries = self.detect_section_headers(full_text)

        chunks = []
        chunk_id = 1

        for i in range(len(boundaries) - 1):
            section_text = full_text[boundaries[i]:boundaries[i+1]]

            # Skip very short sections
            if len(section_text) < self.min_chunk_size:
                continue

            # Split long sections
            if len(section_text) > self.max_chunk_size:
                sub_chunks = self.split_by_paragraphs(section_text, self.target_chunk_size)
            else:
                sub_chunks = [section_text]

            for sub_chunk in sub_chunks:
                if len(sub_chunk) < self.min_chunk_size:
                    continue

                chunk = Chunk(
                    id=f"{doc_name}-chunk{chunk_id:03d}",
                    text=sub_chunk.strip(),
                    doc=doc_name,
                    page_range=self._get_page_range(pages, sub_chunk),
                    topic=self.extract_topic(sub_chunk, doc_name),
                    char_count=len(sub_chunk),
                    sentence_count=self.count_sentences(sub_chunk),
                    keywords=self.extract_keywords(sub_chunk),
                    source_pages=[p['id'] for p in pages]
                )

                chunks.append(chunk)
                chunk_id += 1

        return chunks

    def _get_page_range(self, pages: List[Dict], chunk_text: str) -> List[int]:
        """Determine which pages this chunk spans"""
        # Simple heuristic: find pages that contain chunk text
        page_nums = []
        for page in pages:
            if chunk_text[:100] in page['text'] or chunk_text[-100:] in page['text']:
                page_nums.append(page['page'])
        return page_nums if page_nums else [pages[0]['page']]

    def chunk_dataset(self, jsonl_path: str, output_path: str) -> Dict:
        """
        Process entire dataset and create semantic chunks.

        Args:
            jsonl_path: Path to page-level JSONL file
            output_path: Path to save semantic chunks

        Returns:
            Statistics dictionary
        """
        print(f"\nLoading dataset from: {jsonl_path}")

        # Load dataset
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f]

        print(f"Loaded {len(records)} pages from {len(set(r['doc'] for r in records))} documents")

        # Group by document
        docs = defaultdict(list)
        for record in records:
            docs[record['doc']].append(record)

        # Process each document
        all_chunks = []
        print("\nCreating semantic chunks...")
        for doc_name, pages in sorted(docs.items()):
            doc_chunks = self.chunk_document(pages)
            all_chunks.extend(doc_chunks)
            print(f"  {doc_name:40s}: {len(pages):3d} pages -> {len(doc_chunks):3d} chunks")

        # Save chunks
        print(f"\nSaving chunks to: {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in all_chunks:
                chunk_dict = {
                    'id': chunk.id,
                    'text': chunk.text,
                    'doc': chunk.doc,
                    'page_range': chunk.page_range,
                    'topic': chunk.topic,
                    'char_count': chunk.char_count,
                    'sentence_count': chunk.sentence_count,
                    'keywords': chunk.keywords,
                    'source_pages': chunk.source_pages
                }
                f.write(json.dumps(chunk_dict, ensure_ascii=False) + '\n')

        # Generate statistics
        stats = {
            'total_chunks': len(all_chunks),
            'total_chars': sum(c.char_count for c in all_chunks),
            'avg_chunk_size': sum(c.char_count for c in all_chunks) / len(all_chunks),
            'min_chunk_size': min(c.char_count for c in all_chunks),
            'max_chunk_size': max(c.char_count for c in all_chunks),
            'unique_docs': len(docs),
            'chunks_per_doc': {doc: len([c for c in all_chunks if c.doc == doc])
                               for doc in docs.keys()}
        }

        print(f"\n{'='*60}")
        print(f"Chunking Statistics:")
        print(f"{'='*60}")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Total characters: {stats['total_chars']:,}")
        print(f"Average chunk size: {stats['avg_chunk_size']:.0f} chars")
        print(f"Min chunk size: {stats['min_chunk_size']} chars")
        print(f"Max chunk size: {stats['max_chunk_size']} chars")
        print(f"Unique documents: {stats['unique_docs']}")
        print(f"{'='*60}\n")

        return stats
