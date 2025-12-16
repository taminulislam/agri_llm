#!/usr/bin/env python3
"""
Simple script to generate Q&A batches.

Usage:
    python generate_batch.py --batch-id 1    # Generate batch 1
    python generate_batch.py --batch-id 2    # Generate batch 2
    ...
"""

import json
import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

from qa_generation.core.gemini_client import GeminiClient
from qa_generation.core.quality_validator import QualityValidator
from qa_generation.core.batch_manager import BatchManager
from qa_generation.core.qa_generator import QAGenerator


def create_batch_plans():
    """Create batch plans for all 5 batches"""
    data_dir = Path('data')
    batch_mgr = BatchManager(data_dir)

    chunks_path = data_dir / 'chunks' / 'semantic_chunks.jsonl'

    if not chunks_path.exists():
        print(f"[ERROR] Chunks file not found: {chunks_path}")
        print("\nPlease run first:")
        print("  python qa_generation/scripts/02_create_chunks.py")
        return None

    print("Loading chunks...")
    chunks = [json.loads(line) for line in open(chunks_path, encoding='utf-8')]
    print(f"Loaded {len(chunks)} chunks")

    print("\nCreating batch plan...")
    configs = batch_mgr.create_batch_plan(
        chunks,
        total_target=50000,
        num_batches=5
    )

    for config in configs:
        batch_dir = batch_mgr.save_batch_config(config)
        print(f"  Batch {config.batch_id}: {len(config.chunks_to_process)} chunks -> {config.target_qa_count} Q&A pairs")

    return batch_mgr


def generate_batch(batch_id: int, resume: bool = True):
    """Generate a single batch"""
    # Load environment
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')

    if not api_key:
        print("[ERROR] GEMINI_API_KEY not found in .env file")
        print("\nPlease run first:")
        print("  python qa_generation/scripts/00_setup_guide.py")
        return

    # Initialize components
    data_dir = Path('data')
    batch_mgr = BatchManager(data_dir)
    validator = QualityValidator()
    generator = QAGenerator(api_key, data_dir, batch_mgr, validator)

    # Check if batch config exists
    batch_config_path = data_dir / 'batches' / f'batch_{batch_id:03d}' / 'config.json'
    if not batch_config_path.exists():
        print(f"[INFO] Batch {batch_id} config not found. Creating batch plans...")
        batch_mgr = create_batch_plans()
        if not batch_mgr:
            return

    print(f"\n{'='*80}")
    print(f"GENERATING BATCH {batch_id}")
    print(f"{'='*80}\n")

    # Generate batch
    try:
        stats = generator.generate_batch(batch_id, resume=resume)

        print(f"\n{'='*80}")
        print(f"BATCH {batch_id} COMPLETED!")
        print(f"{'='*80}")
        print(f"\n[RESULTS]")
        print(f"  Generated (raw): {stats['qa_generated']}")
        print(f"  Filtered (kept): {stats['qa_filtered']}")
        print(f"  Rejected: {stats['qa_rejected']}")
        print(f"  Filter pass rate: {stats['filter_rate']}%")
        print(f"  Completion rate: {stats['completion_rate']}% of target")
        print(f"\n[API USAGE]")
        print(f"  Total requests: {stats['api_usage']['total_requests']}")
        print(f"  Success rate: {stats['api_usage']['success_rate']}%")
        print(f"  Estimated cost: ${stats['api_usage']['estimated_cost_usd']}")
        print(f"\n[OUTPUT FILES]")
        batch_dir = data_dir / 'batches' / f'batch_{batch_id:03d}'
        print(f"  Raw Q&A: {batch_dir / 'raw_qa_pairs.jsonl'}")
        print(f"  Filtered Q&A: {batch_dir / 'filtered_qa_pairs.jsonl'}")
        print(f"  Metadata: {batch_dir / 'metadata.json'}")
        print(f"\n{'='*80}\n")

        # Next steps
        if stats['completion_rate'] >= 90:
            print("[OK] Batch generation successful!")
            print("\nNext steps:")
            print(f"  1. Review batch {batch_id}:")
            print(f"     - Check {batch_dir / 'filtered_qa_pairs.jsonl'}")
            print(f"     - Review samples for quality")
            if batch_id < 5:
                print(f"  2. Generate next batch:")
                print(f"     python generate_batch.py --batch-id {batch_id + 1}")
            else:
                print(f"  2. All batches complete! Merge them:")
                print(f"     python merge_batches.py")
        else:
            print("[WARNING] Completion rate is low. You may want to:")
            print("  - Generate more chunks from this batch")
            print("  - Adjust parameters and regenerate")

    except Exception as e:
        print(f"\n[ERROR] Error during batch generation: {e}")
        import traceback
        traceback.print_exc()
        print("\nThe batch can be resumed by running the same command again.")


def main():
    parser = argparse.ArgumentParser(description='Generate Q&A batches')
    parser.add_argument('--batch-id', type=int, required=True, choices=[1, 2, 3, 4, 5],
                        help='Batch number to generate (1-5)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start from beginning (ignore checkpoints)')

    args = parser.parse_args()

    generate_batch(args.batch_id, resume=not args.no_resume)


if __name__ == "__main__":
    main()
