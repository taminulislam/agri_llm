"""
Main Q&A generation orchestrator with parallel processing support.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .gemini_client import GeminiClient
from .quality_validator import QualityValidator
from .batch_manager import BatchManager, BatchCheckpoint
from ..config.prompts import BATCH_GENERATION_PROMPT


class QAGenerator:
    """
    Main orchestrator for Q&A generation from agricultural text chunks.
    """

    # Number of parallel workers for chunk processing
    PARALLEL_WORKERS = 10
    
    def __init__(self,
                 api_key: str,
                 data_dir: Path,
                 batch_manager: BatchManager,
                 quality_validator: QualityValidator):

        self.client = GeminiClient(api_key)
        self.data_dir = Path(data_dir)
        self.batch_manager = batch_manager
        self.validator = quality_validator
        
        # Thread-safe lock for file writing
        self._file_lock = threading.Lock()

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

        # Get chunks to process (excluding already processed)
        chunks_to_process = [
            (idx, chunk_id, all_chunks[chunk_id])
            for idx, chunk_id in enumerate(config.chunks_to_process, 1)
            if chunk_id not in processed_chunks
        ]
        
        self.logger.info(f"Processing {len(chunks_to_process)} chunks in parallel (workers={self.PARALLEL_WORKERS})")
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.PARALLEL_WORKERS) as executor:
            # Submit all chunk processing tasks
            future_to_chunk = {
                executor.submit(
                    self._process_single_chunk, 
                    idx, chunk_id, chunk, config, batch_id
                ): chunk_id
                for idx, chunk_id, chunk in chunks_to_process
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    result = future.result()
                    
                    if result:
                        qa_pairs, accepted, rejected, api_calls = result
                        
                        # Thread-safe file writing
                        with self._file_lock:
                            # Write raw pairs
                            for pair in qa_pairs:
                                raw_file.write(json.dumps(pair, ensure_ascii=False) + '\n')
                                stats['qa_generated'] += 1
                            
                            # Write filtered pairs
                            for pair in accepted:
                                filtered_file.write(json.dumps(pair, ensure_ascii=False) + '\n')
                                stats['qa_filtered'] += 1
                            
                            stats['qa_rejected'] += len(rejected)
                            stats['api_calls'] += api_calls
                            processed_chunks.add(chunk_id)
                            stats['chunks_processed'] += 1
                            
                            # Flush files to ensure data is written
                            raw_file.flush()
                            filtered_file.flush()
                        
                        # Log progress
                        progress = stats['chunks_processed'] / stats['chunks_total'] * 100
                        self.logger.info(f"[{stats['chunks_processed']}/{stats['chunks_total']}] {chunk_id} done - {len(accepted)} Q&A (total: {stats['qa_filtered']})")
                        
                        # Checkpoint periodically
                        if stats['chunks_processed'] % 10 == 0:
                            self._save_checkpoint(batch_id, processed_chunks, stats)
                    
                except Exception as e:
                    self.logger.error(f"Error processing chunk {chunk_id}: {e}")
                    stats['errors'] += 1

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

    def _save_checkpoint(self, batch_id: int, processed_chunks: set, stats: Dict):
        """Save checkpoint for recovery"""
        checkpoint = BatchCheckpoint(
            batch_id=batch_id,
            processed_chunks=list(processed_chunks),
            generated_qa_count=stats['qa_filtered'],
            last_chunk_id=list(processed_chunks)[-1] if processed_chunks else "",
            timestamp=datetime.now().isoformat(),
            api_calls_made=stats['api_calls'],
            cost_so_far=self.client.stats.estimated_cost_usd
        )
        self.batch_manager.save_checkpoint(checkpoint)
        self.logger.info(f"Checkpoint saved: {len(processed_chunks)} chunks, {stats['qa_filtered']} Q&A")

    def _process_single_chunk(self, idx: int, chunk_id: str, chunk: Dict, config, batch_id: int):
        """
        Process a single chunk - designed for parallel execution.
        
        Returns:
            tuple: (qa_pairs, accepted, rejected, api_calls) or None on error
        """
        try:
            self.logger.info(f"Starting chunk {idx}: {chunk_id}")
            
            # Generate Q&A pairs for this chunk (multi-pass with parallel passes)
            qa_pairs, api_calls_made = self._generate_for_chunk_parallel(chunk, config)
            
            # Add metadata to pairs
            for pair in qa_pairs:
                pair['chunk_id'] = chunk_id
                pair['batch_id'] = batch_id
            
            # Quality filtering
            chunk_texts = {chunk_id: chunk['text']}
            accepted, rejected = self.validator.filter_batch(qa_pairs, chunk_texts)
            
            return (qa_pairs, accepted, rejected, api_calls_made)
            
        except Exception as e:
            self.logger.error(f"Error in chunk {chunk_id}: {e}")
            return None

    def _generate_for_chunk_parallel(self, chunk: Dict, config) -> tuple:
        """
        Generate Q&A pairs for a single chunk using PARALLEL passes.
        
        Returns:
            tuple: (list of Q&A pairs, number of API calls made)
        """
        passes = getattr(config, 'passes_per_chunk', 1)
        all_qa_pairs = []
        api_calls_made = 0
        
        # Run all passes in parallel for this chunk
        with ThreadPoolExecutor(max_workers=passes) as executor:
            futures = [
                executor.submit(self._single_pass_generate, chunk, config, pass_num)
                for pass_num in range(passes)
            ]
            
            for future in as_completed(futures):
                try:
                    qa_pairs, pass_num = future.result()
                    all_qa_pairs.extend(qa_pairs)
                    api_calls_made += 1
                    self.logger.info(f"  Pass {pass_num + 1}/{passes}: {len(qa_pairs)} Q&A pairs")
                except Exception as e:
                    self.logger.error(f"  Pass failed: {e}")
                    api_calls_made += 1
        
        return all_qa_pairs, api_calls_made

    def _single_pass_generate(self, chunk: Dict, config, pass_num: int) -> tuple:
        """Generate Q&A pairs for a single pass - thread-safe"""
        variation = self.PASS_VARIATIONS[pass_num % len(self.PASS_VARIATIONS)]
        variation_text = f"\n\nAdditional focus for this batch: {variation}" if variation else ""
        
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
        
        qa_pairs = self.client.generate_qa_batch(prompt)
        
        # Add pass number to each pair
        for pair in qa_pairs:
            pair['pass'] = pass_num + 1
        
        return qa_pairs, pass_num

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
