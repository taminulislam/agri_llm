#!/usr/bin/env python3
"""
Real-time monitoring of batch generation progress.

Shows live updates of Q&A generation progress.
"""

import json
import time
from pathlib import Path
from datetime import datetime


def monitor_batch(batch_id=1):
    """Monitor a running batch generation"""
    data_dir = Path('data')
    batch_dir = data_dir / 'batches' / f'batch_{batch_id:03d}'

    raw_qa_path = batch_dir / 'raw_qa_pairs.jsonl'
    filtered_qa_path = batch_dir / 'filtered_qa_pairs.jsonl'
    checkpoint_path = data_dir / 'checkpoints' / f'batch_{batch_id:03d}_checkpoint.json'

    print(f"\n{'='*80}")
    print(f"MONITORING BATCH {batch_id} - Real-time Progress")
    print(f"{'='*80}\n")
    print("Press Ctrl+C to stop monitoring (generation will continue)\n")

    last_raw_count = 0
    last_filtered_count = 0
    start_time = time.time()

    try:
        while True:
            # Count Q&A pairs
            raw_count = 0
            filtered_count = 0

            if raw_qa_path.exists():
                with open(raw_qa_path, 'r', encoding='utf-8') as f:
                    raw_count = sum(1 for _ in f)

            if filtered_qa_path.exists():
                with open(filtered_qa_path, 'r', encoding='utf-8') as f:
                    filtered_count = sum(1 for _ in f)

            # Load checkpoint info
            checkpoint_info = ""
            if checkpoint_path.exists():
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                    chunks_done = len(checkpoint.get('processed_chunks', []))
                    cost = checkpoint.get('cost_so_far', 0)
                    checkpoint_info = f"Chunks: {chunks_done}/56 | Cost: ${cost:.4f}"

            # Calculate pass rate
            pass_rate = (filtered_count / raw_count * 100) if raw_count > 0 else 0

            # Calculate speed
            elapsed = time.time() - start_time
            qa_per_min = (filtered_count / elapsed * 60) if elapsed > 0 else 0

            # New items since last check
            new_raw = raw_count - last_raw_count
            new_filtered = filtered_count - last_filtered_count

            # Clear line and print status
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Generated: {raw_count:,} | "
                  f"Filtered: {filtered_count:,} | "
                  f"Pass: {pass_rate:.1f}% | "
                  f"Speed: {qa_per_min:.0f}/min | "
                  f"{checkpoint_info}",
                  end='', flush=True)

            if new_raw > 0 or new_filtered > 0:
                print(f" [+{new_filtered}]", end='', flush=True)

            last_raw_count = raw_count
            last_filtered_count = filtered_count

            time.sleep(5)  # Update every 5 seconds

    except KeyboardInterrupt:
        print("\n\n[INFO] Monitoring stopped. Generation continues in background.")
        print(f"\nCurrent status:")
        print(f"  Raw Q&A pairs: {raw_count:,}")
        print(f"  Filtered Q&A pairs: {filtered_count:,}")
        print(f"  Pass rate: {pass_rate:.1f}%")
        print(f"\nTo check final status:")
        print(f"  python monitor_batch.py")
        print(f"\nOutput files:")
        print(f"  {filtered_qa_path}")


if __name__ == "__main__":
    monitor_batch(1)
