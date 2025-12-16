#!/usr/bin/env python3
"""Quick test script to verify Google Gemini API connection."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    print("[ERROR] GEMINI_API_KEY not found in .env file")
    print("\nPlease create a .env file with:")
    print("GEMINI_API_KEY=your_api_key_here")
    exit(1)

print(f"[OK] API Key found: {api_key[:10]}...{api_key[-4:]}")
print("\nTesting API connection...")

try:
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    # Use the current stable model
    model = genai.GenerativeModel('gemini-2.5-flash')

    response = model.generate_content("Say 'Hello! I'm ready to generate agricultural Q&A pairs!'")

    print(f"\n[OK] API test successful!")
    print(f"[OK] Using model: gemini-2.5-flash")
    print(f"\nResponse from Gemini:")
    print(f"  {response.text}")

    print("\n" + "="*80)
    print("SUCCESS! Your Google Gemini API is working!")
    print("="*80)
    print("\nYou're ready to generate Q&A pairs!")
    print("\nNext steps:")
    print("  1. Create chunks: python qa_generation/scripts/02_create_chunks.py")
    print("  2. Generate batch: python generate_batch.py --batch-id 1")
    print("="*80 + "\n")

except Exception as e:
    print(f"\n[ERROR] API test failed: {e}")
    print("\nTroubleshooting:")
    print("1. Verify your API key is correct")
    print("2. Check: https://aistudio.google.com/app/apikey")
    print("3. Make sure Generative Language API is enabled")
    exit(1)
