"""
Main Q&A generation orchestrator.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from .gemini_client import GeminiClient
from .quality_validator import QualityValidator
from .batch_manager import BatchManager, BatchCheckpoint
from ..config.prompts import BATCH_GENERATION_PROMPT


class QAGenerator:
    """
    Main orchestrator for Q&A generation from agricultural text chunks.
    """

    def __init__(self,
                 api_key: str,
                 data_dir: Path,
                 batch_manager: BatchManager,
                 quality_validator: QualityValidator):

        self.client = GeminiClient(api_key)
        self.data_dir = Path(data_dir)
        self.batch_manager = batch_manager
        self.validator = quality_validator

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def generate_batch(self, batch_id: int, resume: bool = True) -> Dict:
        """
        Generate Q&A pairs for a complete batch.

        Args:
            batch_id: Batch number (1-5)
            resume: Whether to resume from checkpoint

        Returns:
            Generation statistics
        """
        self.logger.info(f"Starting batch {batch_id} generation")

        # Load batch configuration
        config = self.batch_manager.load_batch_config(batch_id)
        batch_dir = self.data_dir / "batches" / f"batch_{batch_id:03d}"

        # Check for checkpoint
        checkpoint = self.batch_manager.load_checkpoint(batch_id) if resume else None
        processed_chunks = set(checkpoint.processed_chunks) if checkpoint else set()

        if checkpoint:
            self.logger.info(f"Resuming from checkpoint: {len(processed_chunks)} chunks already processed")

        # Load chunks
        chunks_path = self.data_dir / "chunks" / "semantic_chunks.jsonl"
        all_chunks = {}

        with open(chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                if chunk['id'] in config.chunks_to_process:
                    all_chunks[chunk['id']] = chunk

        self.logger.info(f"Loaded {len(all_chunks)} chunks for processing")

        # Open output files
        raw_qa_path = batch_dir / "raw_qa_pairs.jsonl"
        filtered_qa_path = batch_dir / "filtered_qa_pairs.jsonl"

        raw_file = open(raw_qa_path, 'a', encoding='utf-8')
        filtered_file = open(filtered_qa_path, 'a', encoding='utf-8')

        # Generation stats
        stats = {
            'batch_id': batch_id,
            'chunks_total': len(config.chunks_to_process),
            'chunks_processed': len(processed_chunks),
            'qa_generated': 0,
            'qa_filtered': 0,
            'qa_rejected': 0,
            'api_calls': 0,
            'errors': 0,
            'start_time': datetime.now().isoformat()
        }

        # Process each chunk
        for idx, chunk_id in enumerate(config.chunks_to_process, 1):
            if chunk_id in processed_chunks:
                continue  # Skip already processed

            chunk = all_chunks[chunk_id]

            try:
                self.logger.info(f"Processing chunk {idx}/{len(config.chunks_to_process)}: {chunk_id}")

                # Generate Q&A pairs for this chunk (multi-pass)
                qa_pairs, api_calls_made = self._generate_for_chunk(chunk, config)

                # Write raw pairs
                for pair in qa_pairs:
                    pair['chunk_id'] = chunk_id
                    pair['batch_id'] = batch_id
                    raw_file.write(json.dumps(pair, ensure_ascii=False) + '\n')
                    stats['qa_generated'] += 1

                # Quality filtering
                chunk_texts = {chunk_id: chunk['text']}
                accepted, rejected = self.validator.filter_batch(qa_pairs, chunk_texts)

                # Write filtered pairs
                for pair in accepted:
                    filtered_file.write(json.dumps(pair, ensure_ascii=False) + '\n')
                    stats['qa_filtered'] += 1

                stats['qa_rejected'] += len(rejected)
                stats['api_calls'] += api_calls_made
                processed_chunks.add(chunk_id)
                stats['chunks_processed'] += 1

                # Log progress
                if stats['chunks_processed'] % 10 == 0:
                    progress = stats['chunks_processed'] / stats['chunks_total'] * 100
                    self.logger.info(f"Progress: {progress:.1f}% ({stats['chunks_processed']}/{stats['chunks_total']})")
                    self.logger.info(f"Generated: {stats['qa_generated']}, Filtered: {stats['qa_filtered']} (pass rate: {stats['qa_filtered']/stats['qa_generated']*100:.1f}%)")

                # Checkpoint every N chunks
                if stats['chunks_processed'] % self.batch_manager.checkpoint_frequency == 0:
                    checkpoint = BatchCheckpoint(
                        batch_id=batch_id,
                        processed_chunks=list(processed_chunks),
                        generated_qa_count=stats['qa_filtered'],
                        last_chunk_id=chunk_id,
                        timestamp=datetime.now().isoformat(),
                        api_calls_made=stats['api_calls'],
                        cost_so_far=self.client.stats.estimated_cost_usd
                    )
                    self.batch_manager.save_checkpoint(checkpoint)
                    self.logger.info(f"Checkpoint saved")

            except Exception as e:
                self.logger.error(f"Error processing chunk {chunk_id}: {e}")
                stats['errors'] += 1
                continue

        # Close files
        raw_file.close()
        filtered_file.close()

        # Final statistics
        stats['end_time'] = datetime.now().isoformat()
        stats['completion_rate'] = round(stats['qa_filtered'] / config.target_qa_count * 100, 1)
        stats['filter_rate'] = round(stats['qa_filtered'] / stats['qa_generated'] * 100, 1) if stats['qa_generated'] > 0 else 0
        stats['api_usage'] = self.client.get_usage_stats()

        # Save metadata
        metadata_path = batch_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        self.logger.info(f"Batch {batch_id} generation completed!")
        self.logger.info(f"Generated: {stats['qa_generated']}, Filtered: {stats['qa_filtered']}, Rejected: {stats['qa_rejected']}")

        return stats

    # Pass variation prompts to encourage diverse questions across passes
    PASS_VARIATIONS = [
        "",  # Default - no variation
        "Focus on fundamental concepts and definitions.",
        "Focus on practical applications and real-world scenarios.",
        "Focus on advanced topics and analytical thinking.",
        "Focus on comparisons, trade-offs, and decision-making.",
    ]

    def _generate_for_chunk(self, chunk: Dict, config) -> tuple:
        """
        Generate Q&A pairs for a single chunk using multiple passes.
        
        Returns:
            tuple: (list of Q&A pairs, number of API calls made)
        """
        all_qa_pairs = []
        api_calls_made = 0
        passes = getattr(config, 'passes_per_chunk', 1)
        
        for pass_num in range(passes):
            # Get variation text for this pass
            variation = self.PASS_VARIATIONS[pass_num % len(self.PASS_VARIATIONS)]
            variation_text = f"\n\nAdditional focus for this batch: {variation}" if variation else ""
            
            # Build prompt
            prompt = BATCH_GENERATION_PROMPT.format(
                text=chunk['text'],
                topic=chunk['topic'],
                keywords=', '.join(chunk['keywords']),
                total_questions=config.qa_per_chunk,
                factual_count=config.question_distribution['factual'],
                conceptual_count=config.question_distribution['conceptual'],
                procedural_count=config.question_distribution['procedural'],
                comparative_count=config.question_distribution['comparative'],
                scenario_count=config.question_distribution['scenario'],
                analytical_count=config.question_distribution['analytical']
            ) + variation_text

            # Generate Q&A pairs for this pass
            qa_pairs = self.client.generate_qa_batch(prompt)
            api_calls_made += 1
            
            # Add pass number to each pair for tracking
            for pair in qa_pairs:
                pair['pass'] = pass_num + 1
            
            all_qa_pairs.extend(qa_pairs)
            
            # Log progress for multi-pass
            if passes > 1:
                self.logger.info(f"  Pass {pass_num + 1}/{passes}: generated {len(qa_pairs)} Q&A pairs")

        return all_qa_pairs, api_calls_made
