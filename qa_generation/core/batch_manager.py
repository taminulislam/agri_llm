"""
Batch processing manager with checkpointing and recovery.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class BatchConfig:
    """Configuration for a single batch"""
    batch_id: int
    target_qa_count: int  # e.g., 10,000
    chunks_to_process: List[str]  # chunk IDs
    qa_per_chunk: int  # questions per single API call (max 20 for reliability)
    passes_per_chunk: int  # number of API calls per chunk (e.g., 5)
    question_distribution: Dict[str, int]  # type -> count (per single API call)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed


@dataclass
class BatchCheckpoint:
    """Checkpoint for resuming generation"""
    batch_id: int
    processed_chunks: List[str]
    generated_qa_count: int
    last_chunk_id: str
    timestamp: str
    api_calls_made: int
    cost_so_far: float


class BatchManager:
    """
    Manage batch processing with checkpointing and recovery.

    Features:
    - Splits dataset into batches
    - Tracks progress with checkpoints
    - Enables resume after interruption
    - Generates batch reports
    """

    def __init__(self,
                 data_dir: Path,
                 batch_size: int = 10000,
                 checkpoint_frequency: int = 50):  # Save every 50 chunks

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.checkpoint_frequency = checkpoint_frequency

        # Create directories
        self.batches_dir = self.data_dir / "batches"
        self.checkpoints_dir = self.data_dir / "checkpoints"
        self.batches_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Maximum Q&A pairs per single API call (to avoid JSON truncation)
    MAX_QA_PER_CALL = 20
    
    def create_batch_plan(self,
                          chunks: List[Dict],
                          total_target: int = 50000,
                          num_batches: int = 5) -> List[BatchConfig]:
        """
        Create batch processing plan.

        Args:
            chunks: List of semantic chunks
            total_target: Total Q&A pairs to generate (50,000)
            num_batches: Number of batches (5)

        Returns:
            List of batch configurations
        """
        qa_per_batch = total_target // num_batches
        chunks_per_batch = len(chunks) // num_batches

        batches = []

        for batch_num in range(1, num_batches + 1):
            # Calculate chunk range for this batch
            start_idx = (batch_num - 1) * chunks_per_batch
            end_idx = start_idx + chunks_per_batch if batch_num < num_batches else len(chunks)

            batch_chunks = chunks[start_idx:end_idx]
            chunk_ids = [c['id'] for c in batch_chunks]

            # Calculate total questions needed per chunk
            total_qa_per_chunk = max(1, qa_per_batch // len(batch_chunks))
            
            # Calculate passes needed per chunk (each pass generates MAX_QA_PER_CALL questions)
            # Use 20 Q&A per API call to avoid JSON truncation issues
            qa_per_call = min(self.MAX_QA_PER_CALL, total_qa_per_chunk)
            passes_per_chunk = max(1, (total_qa_per_chunk + qa_per_call - 1) // qa_per_call)
            
            # Cap passes to avoid excessive API calls (5 passes max = 100 Q&A per chunk)
            passes_per_chunk = min(passes_per_chunk, 5)
            
            # Recalculate actual Q&A per chunk based on passes
            actual_qa_per_chunk = qa_per_call * passes_per_chunk

            # Define question type distribution (as counts per single API call)
            distribution = {
                'factual': max(1, int(qa_per_call * 0.20)),
                'conceptual': max(1, int(qa_per_call * 0.20)),
                'procedural': max(1, int(qa_per_call * 0.20)),
                'comparative': max(1, int(qa_per_call * 0.15)),
                'scenario': max(1, int(qa_per_call * 0.15)),
                'analytical': max(1, int(qa_per_call * 0.10)),
            }

            # Ensure sum equals qa_per_call
            actual_sum = sum(distribution.values())
            if actual_sum < qa_per_call:
                distribution['factual'] += (qa_per_call - actual_sum)
            elif actual_sum > qa_per_call:
                # Trim excess from factual
                distribution['factual'] -= (actual_sum - qa_per_call)

            config = BatchConfig(
                batch_id=batch_num,
                target_qa_count=len(batch_chunks) * actual_qa_per_chunk,  # Realistic target
                chunks_to_process=chunk_ids,
                qa_per_chunk=qa_per_call,  # Per single API call
                passes_per_chunk=passes_per_chunk,  # Number of API calls per chunk
                question_distribution=distribution
            )

            batches.append(config)

        return batches

    def save_batch_config(self, config: BatchConfig) -> Path:
        """Save batch configuration"""
        batch_dir = self.batches_dir / f"batch_{config.batch_id:03d}"
        batch_dir.mkdir(exist_ok=True)

        config_path = batch_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(config), f, indent=2)

        return batch_dir

    def load_batch_config(self, batch_id: int) -> BatchConfig:
        """Load batch configuration"""
        config_path = self.batches_dir / f"batch_{batch_id:03d}" / "config.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return BatchConfig(**data)

    def save_checkpoint(self, checkpoint: BatchCheckpoint):
        """Save checkpoint for resuming"""
        checkpoint_path = self.checkpoints_dir / f"batch_{checkpoint.batch_id:03d}_checkpoint.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(checkpoint), f, indent=2)

    def load_checkpoint(self, batch_id: int) -> Optional[BatchCheckpoint]:
        """Load checkpoint if exists"""
        checkpoint_path = self.checkpoints_dir / f"batch_{batch_id:03d}_checkpoint.json"
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return BatchCheckpoint(**data)
        return None

    def get_batch_progress(self, batch_id: int) -> Dict:
        """Get progress statistics for a batch"""
        batch_dir = self.batches_dir / f"batch_{batch_id:03d}"

        if not batch_dir.exists():
            return {"status": "not_started"}

        # Load config
        config = self.load_batch_config(batch_id)

        # Check for checkpoint
        checkpoint = self.load_checkpoint(batch_id)

        # Count generated Q&A pairs
        raw_qa_path = batch_dir / "raw_qa_pairs.jsonl"
        filtered_qa_path = batch_dir / "filtered_qa_pairs.jsonl"

        raw_count = 0
        filtered_count = 0

        if raw_qa_path.exists():
            with open(raw_qa_path, 'r', encoding='utf-8') as f:
                raw_count = sum(1 for _ in f)

        if filtered_qa_path.exists():
            with open(filtered_qa_path, 'r', encoding='utf-8') as f:
                filtered_count = sum(1 for _ in f)

        progress_pct = (filtered_count / config.target_qa_count * 100) if config.target_qa_count > 0 else 0

        return {
            "batch_id": batch_id,
            "status": config.status,
            "target_qa_count": config.target_qa_count,
            "raw_generated": raw_count,
            "filtered_count": filtered_count,
            "progress_percentage": round(progress_pct, 1),
            "chunks_total": len(config.chunks_to_process),
            "chunks_processed": len(checkpoint.processed_chunks) if checkpoint else 0,
            "last_checkpoint": checkpoint.timestamp if checkpoint else None,
            "cost_so_far": checkpoint.cost_so_far if checkpoint else 0.0
        }
