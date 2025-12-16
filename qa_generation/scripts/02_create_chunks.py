#!/usr/bin/env python3
"""
Create semantic chunks from agricultural dataset.

This script processes the page-level JSONL dataset and creates
topic-coherent chunks optimized for Q&A generation.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qa_generation.core.chunker import SemanticChunker


def main():
    print("\n" + "="*80)
    print("AGRICULTURAL DATASET - SEMANTIC CHUNKING")
    print("="*80 + "\n")

    # Paths
    project_root = Path(__file__).parent.parent.parent
    input_path = project_root / "extracted_data" / "agricultural_dataset.jsonl"
    output_path = project_root / "data" / "chunks" / "semantic_chunks.jsonl"

    # Check if input exists
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        print("\nPlease make sure you have the agricultural_dataset.jsonl file in:")
        print(f"  {input_path.parent}/")
        return

    # Create chunker
    print("Configuration:")
    print("  Target chunk size: 2000 characters")
    print("  Min chunk size: 1000 characters")
    print("  Max chunk size: 3500 characters")
    print("  Overlap: 200 characters\n")

    chunker = SemanticChunker(
        target_chunk_size=2000,
        min_chunk_size=1000,
        max_chunk_size=3500,
        overlap_size=200
    )

    # Process dataset
    try:
        stats = chunker.chunk_dataset(str(input_path), str(output_path))

        print("\n[OK] Chunking completed successfully!")
        print(f"\nOutput saved to: {output_path}")

        # Display recommendations
        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("\n1. Review the chunks (optional):")
        print(f"   - Open: {output_path}")
        print("   - Check that chunks are coherent and well-sized")
        print("\n2. Generate batch plan:")
        print("   python generate_batch.py --batch-id 1")
        print("\n3. Generate first batch:")
        print("   python generate_batch.py --batch-id 1")
        print("\n" + "="*80 + "\n")

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("\nPlease check that the input file exists and is accessible.")
    except Exception as e:
        print(f"[ERROR] Error during chunking: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
