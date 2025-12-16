# Quick Start Guide - Q&A Dataset Generation

Generate 50,000+ agricultural Q&A pairs in just a few steps!

## Prerequisites

- Python 3.8+
- Your agricultural dataset at `extracted_data/agricultural_dataset.jsonl` ✅ (you have this)
- Google account (free)

## Step-by-Step Guide

### Step 1: Install Dependencies (2 minutes)

```bash
cd d:\github_repos\agri_llm
pip install -r requirements.txt
```

This installs:
- `google-generativeai` - Gemini API client
- `nltk` - Text processing
- `tenacity` - Retry logic
- `python-dotenv` - Environment variables
- `tqdm` - Progress bars

### Step 2: Set Up Google Gemini API (5-10 minutes)

Run the interactive setup wizard:

```bash
python qa_generation/scripts/00_setup_guide.py
```

**What it does:**
1. Guides you to create a Google Cloud account (if needed)
2. Shows you how to enable the Generative Language API
3. Helps you create an API key
4. Saves the API key to `.env` file
5. Tests the API connection
6. Adds `.env` to `.gitignore` for security

**Free Tier Limits:**
- 15 requests/minute
- 1M tokens/minute
- 1,500 requests/day
- **This covers ~80% of your 50k Q&A generation project!**

### Step 3: Create Semantic Chunks (2-3 minutes)

Transform your 249 pages into optimized chunks:

```bash
python qa_generation/scripts/02_create_chunks.py
```

**What it does:**
- Loads `extracted_data/agricultural_dataset.jsonl`
- Groups pages by document
- Detects section boundaries
- Creates 400-600 coherent chunks (1500-2500 chars each)
- Saves to `data/chunks/semantic_chunks.jsonl`

**Expected output:**
```
Chunking Statistics:
Total chunks: 486
Total characters: 967,037
Average chunk size: 1,989 chars
Min chunk size: 1,012 chars
Max chunk size: 3,487 chars
Unique documents: 14
```

### Step 4: Generate First Batch (2-3 hours)

Generate your first 10,000 Q&A pairs:

```bash
python generate_batch.py --batch-id 1
```

**What it does:**
- Creates batch plan (5 batches × 10k Q&A pairs)
- Processes ~97 chunks
- Generates ~100 Q&A pairs per chunk
- Applies quality filtering (70-80% pass rate)
- Saves checkpoints every 50 chunks
- Shows real-time progress

**While it runs, you'll see:**
```
Processing chunk 10/97: chapter02-chunk003
Progress: 10.3% (10/97)
Generated: 1000, Filtered: 756 (75.6% pass rate)
```

**Output files:**
- `data/batches/batch_001/raw_qa_pairs.jsonl` - All generated
- `data/batches/batch_001/filtered_qa_pairs.jsonl` - Quality-filtered (~10k)
- `data/batches/batch_001/metadata.json` - Statistics

### Step 5: Review Batch (5-10 minutes)

Inspect the generated Q&A pairs:

```bash
# Open the filtered file
code data/batches/batch_001/filtered_qa_pairs.jsonl
```

Or use Python:

```python
import json

# Load first 10 samples
with open('data/batches/batch_001/filtered_qa_pairs.jsonl') as f:
    samples = [json.loads(line) for line in f][:10]

for i, qa in enumerate(samples, 1):
    print(f"\n--- Sample {i} ---")
    print(f"Type: {qa['type']}, Difficulty: {qa['difficulty']}")
    print(f"Q: {qa['question']}")
    print(f"A: {qa['answer']}")
    print(f"Quality: {qa['quality_score']['overall']:.1f}/10")
```

**Check for:**
- ✅ Questions are clear and answerable
- ✅ Answers are factually accurate
- ✅ Good variety in question types
- ✅ Appropriate difficulty distribution

### Step 6: Generate Remaining Batches (1-2 days)

Continue with batches 2-5:

```bash
python generate_batch.py --batch-id 2  # ~2-3 hours
python generate_batch.py --batch-id 3  # ~2-3 hours
python generate_batch.py --batch-id 4  # ~2-3 hours
python generate_batch.py --batch-id 5  # ~2-3 hours
```

**💡 Tip:** You can pause between batches to review quality!

**If interrupted:** The script automatically checkpoints every 50 chunks. Just re-run the same command to resume:

```bash
python generate_batch.py --batch-id 2  # Automatically resumes from checkpoint
```

### Step 7: Merge Final Dataset (2 minutes)

Once all batches are approved, merge them:

```python
# merge_batches.py
import json
from pathlib import Path

def merge_batches():
    all_qa = []
    seen_questions = set()

    for batch_id in range(1, 6):
        batch_file = Path(f'data/batches/batch_{batch_id:03d}/filtered_qa_pairs.jsonl')

        with open(batch_file, 'r', encoding='utf-8') as f:
            for line in f:
                qa = json.loads(line)
                q_norm = qa['question'].lower().strip()

                if q_norm not in seen_questions:
                    seen_questions.add(q_norm)
                    all_qa.append(qa)

    # Save merged dataset
    output_path = Path('data/final/agricultural_qa_dataset.jsonl')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for qa in all_qa:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')

    print(f"✅ Merged dataset saved: {output_path}")
    print(f"   Total Q&A pairs: {len(all_qa):,}")
    print(f"   Duplicates removed: {sum(1 for _ in open(batch_file)) - len(all_qa)}")

if __name__ == "__main__":
    merge_batches()
```

Run it:
```bash
python merge_batches.py
```

## Final Output

Your final dataset will be at: `data/final/agricultural_qa_dataset.jsonl`

**Format:**
```json
{
  "instruction": "What is the optimal planting depth for corn?",
  "output": "For most conditions, corn should be planted 1.5 to 1.75 inches deep...",
  "difficulty": "basic",
  "type": "factual",
  "chunk_id": "chapter02-chunk003",
  "batch_id": 1,
  "quality_score": { "overall": 8.2, ... }
}
```

**Ready for:**
- ✅ LLM fine-tuning (Llama, GPT, Claude, etc.)
- ✅ RAG systems
- ✅ Question-answering models
- ✅ Educational applications

## Cost Summary

**Estimated total cost:** $1-2

**Breakdown:**
- Batch 1: ~$0.38 (or free tier)
- Batch 2: ~$0.38 (or free tier)
- Batch 3: ~$0.38 (or free tier)
- Batch 4: ~$0.38 (may use paid tier)
- Batch 5: ~$0.38 (may use paid tier)

**Free tier covers:** ~80% (batches 1-3)
**Paid tier:** ~20% (batches 4-5) = ~$0.76

## Troubleshooting

### "GEMINI_API_KEY not found"
→ Run `python qa_generation/scripts/00_setup_guide.py` again

### "Chunks file not found"
→ Run `python qa_generation/scripts/02_create_chunks.py` first

### "API rate limit exceeded"
→ Script automatically waits. Just let it run.

### "Low quality pass rate (<50%)"
→ Check source text quality
→ Review prompt templates in `qa_generation/config/prompts.py`

### "Generation failed mid-batch"
→ Just re-run the same command - it will resume from checkpoint

## What's Next?

After generating your dataset:

1. **Fine-tune an LLM:**
   ```bash
   # Example with Hugging Face
   from datasets import load_dataset
   dataset = load_dataset('json', data_files='data/final/agricultural_qa_dataset.jsonl')
   ```

2. **Build a RAG system:**
   - Use the Q&A pairs as training data
   - Create vector embeddings
   - Build an agricultural chatbot

3. **Create training/validation splits:**
   ```python
   from sklearn.model_selection import train_test_split
   # 80/10/10 split
   train, temp = train_test_split(data, test_size=0.2)
   val, test = train_test_split(temp, test_size=0.5)
   ```

## Support

- **Detailed plan:** See `.claude/plans/moonlit-watching-floyd.md`
- **Full README:** See `README_QA_GENERATION.md`
- **Google Gemini docs:** https://ai.google.dev/docs

---

🎉 **Congratulations! You're ready to generate your 50k Q&A dataset!**
