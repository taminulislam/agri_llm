# Agricultural Q&A Dataset Generation

Generate 50,000+ high-quality question-answer pairs from agricultural text data using Google Gemini Flash API.

## Overview

This project transforms 249 pages of agricultural textbook content into a comprehensive Q&A dataset through:

- **Semantic Chunking**: Converts pages into 400-600 topic-coherent chunks
- **Diverse Q&A Generation**: 6 question types × 3 difficulty levels
- **Quality Filtering**: Automated validation with 70-80% pass rate
- **Batch Processing**: 5 batches of 10k Q&A pairs with manual review
- **Cost-Effective**: ~$1-2 total cost (mostly free tier)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Google Gemini API

Run the interactive setup guide:

```bash
python qa_generation/scripts/00_setup_guide.py
```

This will guide you through:
- Creating a Google Cloud account
- Enabling the Generative Language API
- Creating an API key
- Testing the connection

**Your API key will be saved to `.env` (automatically added to `.gitignore`)**

### 3. Create Semantic Chunks

Process your agricultural dataset into semantic chunks:

```bash
python qa_generation/scripts/02_create_chunks.py
```

**Input**: `extracted_data/agricultural_dataset.jsonl` (249 pages)
**Output**: `data/chunks/semantic_chunks.jsonl` (400-600 chunks)

### 4. Generate Q&A Batches

Create your first batch of 10,000 Q&A pairs:

```bash
python -c "from pathlib import Path; import sys; import json; from dotenv import load_dotenv; import os; sys.path.insert(0, str(Path.cwd())); load_dotenv(); from qa_generation.core.gemini_client import GeminiClient; from qa_generation.core.quality_validator import QualityValidator; from qa_generation.core.batch_manager import BatchManager; from qa_generation.core.qa_generator import QAGenerator; data_dir = Path('data'); batch_mgr = BatchManager(data_dir); chunks = [json.loads(line) for line in open('data/chunks/semantic_chunks.jsonl')]; configs = batch_mgr.create_batch_plan(chunks, 50000, 5); [batch_mgr.save_batch_config(c) for c in configs]; validator = QualityValidator(); generator = QAGenerator(os.getenv('GEMINI_API_KEY'), data_dir, batch_mgr, validator); stats = generator.generate_batch(1); print(json.dumps(stats, indent=2))"
```

**For easier use, you can create a simple script** (see examples below).

## Project Structure

```
agri_llm/
├── qa_generation/
│   ├── config/
│   │   └── prompts.py              # Question type templates
│   ├── core/
│   │   ├── chunker.py              # Semantic text chunking
│   │   ├── gemini_client.py        # API client with rate limiting
│   │   ├── qa_generator.py         # Main orchestrator
│   │   ├── quality_validator.py    # Quality filtering
│   │   └── batch_manager.py        # Batch processing
│   └── scripts/
│       ├── 00_setup_guide.py       # API setup wizard
│       └── 02_create_chunks.py     # Create chunks
├── data/
│   ├── chunks/
│   │   └── semantic_chunks.jsonl   # Processed chunks
│   ├── batches/
│   │   ├── batch_001/
│   │   │   ├── config.json
│   │   │   ├── raw_qa_pairs.jsonl
│   │   │   ├── filtered_qa_pairs.jsonl
│   │   │   └── metadata.json
│   │   └── batch_002/ ... batch_005/
│   └── final/
│       └── agricultural_qa_dataset.jsonl
├── extracted_data/
│   └── agricultural_dataset.jsonl  # Source data (249 pages)
├── .env                             # API key (gitignored)
└── requirements.txt
```

## Question Types & Distribution

The system generates 6 types of questions:

| Type | Distribution | Description | Example |
|------|-------------|-------------|---------|
| **Factual** | 20% | What, When, Where, Define | "What is the optimal planting depth for corn?" |
| **Conceptual** | 20% | Why, How does, Explain | "Why does late planting reduce corn yield?" |
| **Procedural** | 20% | How to, Steps | "How do you calculate growing degree days?" |
| **Comparative** | 15% | Compare, Difference | "What's the difference between flex-ear and fixed-ear hybrids?" |
| **Scenario** | 15% | What if, Application | "If a farmer observes weak stalks, what should they do?" |
| **Analytical** | 10% | Evaluate, Assess | "Evaluate whether to plant full-season or medium-maturity hybrids" |

**Difficulty Levels**:
- Basic (30%): Direct recall from text
- Intermediate (50%): Understanding and connections
- Advanced (20%): Synthesis and critical thinking

## Workflow

### Phase 1: Setup (5-10 minutes)
1. Install dependencies
2. Set up Google Gemini API
3. Test API connection

### Phase 2: Preprocessing (2-3 minutes)
1. Create semantic chunks from 249 pages
2. Verify chunk quality and distribution

### Phase 3: Batch Generation (1-2 days)
For each batch (1-5):
1. **Generate**: Run batch generation (~2-3 hours)
   - Processes ~100 chunks
   - Generates ~10,000 Q&A pairs
   - Automatic quality filtering
   - Checkpointing every 50 chunks
2. **Review**: Manual quality check (~30 minutes)
   - View statistics and samples
   - Approve, reject, or regenerate
3. **Repeat**: Move to next batch

### Phase 4: Final Assembly (10 minutes)
1. Merge all approved batches
2. Deduplicate across batches
3. Generate final statistics
4. Export in multiple formats

## Cost & Time Estimates

### API Costs

**Gemini Flash Free Tier**:
- 15 requests per minute
- 1 million tokens per minute
- 1,500 requests per day
- **Covers ~80% of the project**

**Paid Pricing** (after free tier):
- Input: $0.075 per 1M tokens
- Output: $0.30 per 1M tokens
- **Total project cost: ~$1-2**

### Time Estimates

| Phase | Time |
|-------|------|
| Setup & chunking | 10-15 min |
| Batch 1 generation | 2-3 hours |
| Batch 1 review | 30 min |
| Batches 2-5 | 12-15 hours |
| Final merge | 10 min |
| **Total** | **1-2 days** |

## Quality Assurance

### Automated Filtering

Each Q&A pair is validated through:
- **Length checks**: Q: 10-300 chars, A: 20-1000 chars
- **Format checks**: Questions end with '?', no artifacts
- **Vagueness detection**: Avoid yes/no, too-short answers
- **Deduplication**: Track and remove duplicates
- **Source alignment**: Verify answer grounded in text
- **Quality scoring**: 0-10 scale across 5 dimensions

**Target acceptance rate**: 70-80%

### Manual Review

For each batch:
- View 10+ random samples
- Check statistics and distributions
- Verify factual accuracy
- Approve or request regeneration

## Output Format

Standard instruction format ready for fine-tuning:

```json
{
  "instruction": "What is the optimal planting depth for corn in early season?",
  "output": "For most conditions, corn should be planted 1.5 to 1.75 inches deep. Early-planted corn should be in the shallower end of this range, keeping in mind that variation in depth means some seeds will end up shallower than average.",
  "difficulty": "basic",
  "type": "factual",
  "chunk_id": "chapter02-chunk003",
  "batch_id": 1,
  "quality_score": {
    "overall": 8.2,
    "accuracy": 9.0,
    "clarity": 8.5,
    "completeness": 7.8,
    "relevance": 8.5,
    "specificity": 7.5
  }
}
```

## Example: Simple Generation Script

Create `generate_batch.py`:

```python
#!/usr/bin/env python3
import json
import os
from pathlib import Path
from dotenv import load_dotenv

from qa_generation.core.gemini_client import GeminiClient
from qa_generation.core.quality_validator import QualityValidator
from qa_generation.core.batch_manager import BatchManager
from qa_generation.core.qa_generator import QAGenerator

def main():
    # Load environment
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')

    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file")
        print("Run: python qa_generation/scripts/00_setup_guide.py")
        return

    # Initialize components
    data_dir = Path('data')
    batch_mgr = BatchManager(data_dir)
    validator = QualityValidator()

    # Create batch plan (first time only)
    chunks_path = data_dir / 'chunks' / 'semantic_chunks.jsonl'
    if chunks_path.exists():
        chunks = [json.loads(line) for line in open(chunks_path)]
        configs = batch_mgr.create_batch_plan(chunks, total_target=50000, num_batches=5)
        for config in configs:
            batch_mgr.save_batch_config(config)
        print(f"Created plan for {len(configs)} batches")

    # Generate batch
    batch_id = 1
    generator = QAGenerator(api_key, data_dir, batch_mgr, validator)

    print(f"\nGenerating batch {batch_id}...")
    stats = generator.generate_batch(batch_id, resume=True)

    print(f"\nBatch {batch_id} completed!")
    print(f"  Generated: {stats['qa_generated']}")
    print(f"  Filtered: {stats['qa_filtered']}")
    print(f"  Filter rate: {stats['filter_rate']}%")
    print(f"  Cost: ${stats['api_usage']['estimated_cost_usd']}")

if __name__ == "__main__":
    main()
```

Run with:
```bash
python generate_batch.py
```

## Troubleshooting

### API Key Issues
- Ensure API key starts with 'AIza'
- Check that Generative Language API is enabled
- Verify billing is set up (if past free tier)

### Rate Limiting
- Free tier: 15 requests/minute
- Script automatically sleeps when hitting limits
- Progress saved every 50 chunks

### Low Quality Rate
- Review prompt templates in `qa_generation/config/prompts.py`
- Adjust quality thresholds in `QualityValidator`
- Check source text quality

### Generation Errors
- Check logs in `logs/generation.log`
- Review checkpoint files for resume
- Verify chunk file exists and is valid

## Advanced Usage

### Resume from Checkpoint

Batches automatically checkpoint every 50 chunks. To resume:

```python
stats = generator.generate_batch(batch_id=1, resume=True)
```

### Adjust Quality Thresholds

```python
validator = QualityValidator(
    min_question_length=15,  # Stricter
    min_answer_length=30,
    min_overall_score=7.0    # Higher threshold
)
```

### Customize Question Distribution

Edit batch config before generation:

```python
config.question_distribution = {
    'factual': 25,
    'conceptual': 25,
    'procedural': 20,
    'comparative': 10,
    'scenario': 10,
    'analytical': 10,
}
```

## Success Criteria

✅ **Dataset Size**: 50,000+ unique Q&A pairs
✅ **Quality**: Average score ≥7.5/10
✅ **Diversity**: 6 types, 3 difficulty levels
✅ **Cost**: <$3 total
✅ **Format**: Standard instruction format
✅ **Source Alignment**: >95% grounded in text

## Support

For issues or questions:
1. Check this README
2. Review the detailed plan in `.claude/plans/moonlit-watching-floyd.md`
3. Check script help messages
4. Review Google Gemini API documentation

## License

This Q&A generation system is provided as-is for agricultural education and research purposes.
