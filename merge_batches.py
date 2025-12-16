#!/usr/bin/env python3
"""
Merge approved batches into final dataset.

This script:
1. Loads filtered Q&A pairs from all batches
2. Deduplicates across batches
3. Generates statistics
4. Saves final merged dataset
"""

import json
from pathlib import Path
from collections import Counter


def merge_batches(batch_ids=None):
    """
    Merge multiple batches into final dataset.

    Args:
        batch_ids: List of batch IDs to merge (default: [1,2,3,4,5])
    """
    if batch_ids is None:
        batch_ids = [1, 2, 3, 4, 5]

    print(f"\n{'='*80}")
    print("MERGING Q&A BATCHES")
    print(f"{'='*80}\n")

    all_qa = []
    seen_questions = set()
    duplicate_count = 0
    batch_stats = {}

    # Process each batch
    for batch_id in batch_ids:
        batch_file = Path(f'data/batches/batch_{batch_id:03d}/filtered_qa_pairs.jsonl')

        if not batch_file.exists():
            print(f"⚠️  Warning: Batch {batch_id} not found at {batch_file}")
            print(f"   Skipping batch {batch_id}")
            continue

        print(f"Loading batch {batch_id}...")
        batch_qa = []

        with open(batch_file, 'r', encoding='utf-8') as f:
            for line in f:
                qa = json.loads(line)
                batch_qa.append(qa)

                # Check for duplicates
                q_norm = qa['question'].lower().strip()
                q_norm = ''.join(c for c in q_norm if c.isalnum() or c.isspace())

                if q_norm not in seen_questions:
                    seen_questions.add(q_norm)
                    all_qa.append(qa)
                else:
                    duplicate_count += 1

        batch_stats[batch_id] = {
            'total': len(batch_qa),
            'unique': len(batch_qa) - duplicate_count
        }

        print(f"  ✅ Loaded {len(batch_qa)} Q&A pairs")

    print(f"\n{'='*80}")
    print("DEDUPLICATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total Q&A pairs loaded: {sum(s['total'] for s in batch_stats.values()):,}")
    print(f"Duplicates removed: {duplicate_count:,}")
    print(f"Unique Q&A pairs: {len(all_qa):,}")

    # Save merged dataset
    output_path = Path('data/final/agricultural_qa_dataset.jsonl')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for qa in all_qa:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')

    print(f"\n{'='*80}")
    print("DATASET STATISTICS")
    print(f"{'='*80}")

    # Calculate statistics
    types = Counter(qa['type'] for qa in all_qa)
    difficulties = Counter(qa['difficulty'] for qa in all_qa)
    quality_scores = [qa['quality_score']['overall'] for qa in all_qa if 'quality_score' in qa]

    print(f"\nTotal Q&A pairs: {len(all_qa):,}")

    print(f"\nQuestion Type Distribution:")
    for qtype, count in types.most_common():
        pct = count / len(all_qa) * 100
        print(f"  {qtype:15s}: {count:6,} ({pct:5.1f}%)")

    print(f"\nDifficulty Distribution:")
    for diff, count in difficulties.most_common():
        pct = count / len(all_qa) * 100
        print(f"  {diff:15s}: {count:6,} ({pct:5.1f}%)")

    if quality_scores:
        avg_score = sum(quality_scores) / len(quality_scores)
        print(f"\nAverage Quality Score: {avg_score:.2f}/10")
        print(f"Min Score: {min(quality_scores):.2f}/10")
        print(f"Max Score: {max(quality_scores):.2f}/10")

    # Count by batch
    batch_counts = Counter(qa['batch_id'] for qa in all_qa)
    print(f"\nQ&A Pairs by Batch:")
    for batch_id in sorted(batch_counts.keys()):
        count = batch_counts[batch_id]
        pct = count / len(all_qa) * 100
        print(f"  Batch {batch_id}: {count:6,} ({pct:5.1f}%)")

    # Save statistics
    stats = {
        'total_qa_pairs': len(all_qa),
        'duplicates_removed': duplicate_count,
        'type_distribution': dict(types),
        'difficulty_distribution': dict(difficulties),
        'avg_quality_score': avg_score if quality_scores else 0,
        'batch_distribution': dict(batch_counts)
    }

    stats_path = output_path.parent / 'dataset_statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*80}")
    print("OUTPUT FILES")
    print(f"{'='*80}")
    print(f"\n✅ Final dataset: {output_path}")
    print(f"   {len(all_qa):,} unique Q&A pairs")
    print(f"\n✅ Statistics: {stats_path}")

    # Create pretty JSON version
    json_path = output_path.parent / 'agricultural_qa_dataset.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_qa, f, indent=2, ensure_ascii=False)
    print(f"\n✅ JSON format: {json_path}")
    print(f"   (Pretty-printed for easy viewing)")

    print(f"\n{'='*80}")
    print("SUCCESS!")
    print(f"{'='*80}")
    print(f"\nYour {len(all_qa):,} Q&A pairs are ready for:")
    print("  ✅ LLM fine-tuning")
    print("  ✅ RAG systems")
    print("  ✅ Question-answering models")
    print("  ✅ Educational applications")
    print(f"\n{'='*80}\n")

    return all_qa


if __name__ == "__main__":
    merge_batches()
