import json

def convert_to_chatml(input_file, output_file):
    """
    Convert agricultural QA dataset to ChatML format.

    Args:
        input_file: Path to input JSONL or JSON file
        output_file: Path to output JSONL file in ChatML format
    """

    # System prompt for agricultural expert
    system_prompt = "You are an agricultural expert with extensive knowledge about farming, crop production, soil management, and agricultural practices. Provide accurate, detailed, and helpful answers to questions about agriculture."

    # Read input data
    if input_file.endswith('.jsonl'):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    else:  # .json
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

    # Convert to ChatML format
    chatml_data = []
    for item in data:
        chatml_item = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": item["question"]
                },
                {
                    "role": "assistant",
                    "content": item["answer"]
                }
            ]
        }
        chatml_data.append(chatml_item)

    # Write to output file in JSONL format
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in chatml_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"[OK] Converted {len(chatml_data)} items to ChatML format")
    print(f"[OK] Saved to: {output_file}")

    # Also save as regular JSON for inspection
    json_output = output_file.replace('.jsonl', '_formatted.json')
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(chatml_data, f, indent=2, ensure_ascii=False)

    print(f"[OK] Also saved formatted version to: {json_output}")

    # Show sample
    print("\n--- Sample ChatML Entry ---")
    print(json.dumps(chatml_data[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    # Convert the JSONL file (most common format for training)
    input_file = r"D:\github_repos\agri_llm\data\final\agricultural_qa_dataset.jsonl"
    output_file = r"D:\github_repos\agri_llm\data\final\agricultural_qa_chatml.jsonl"

    convert_to_chatml(input_file, output_file)
