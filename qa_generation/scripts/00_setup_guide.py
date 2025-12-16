#!/usr/bin/env python3
"""
Interactive setup guide for Google Gemini API.

Guides user through:
1. Creating Google Cloud account
2. Enabling Gemini API
3. Creating API key
4. Setting up environment variables
"""

import os
from pathlib import Path


def setup_guide():
    print("=" * 80)
    print("GOOGLE GEMINI API SETUP GUIDE")
    print("=" * 80)

    print("\nThis guide will help you set up the Google Gemini API for Q&A generation.")
    print("The process takes about 5-10 minutes.\n")

    print("=" * 80)
    print("Step 1: Create Google Cloud Account")
    print("=" * 80)
    print("\n1. Go to: https://console.cloud.google.com/")
    print("2. Sign in with your Google account (or create one if needed)")
    print("3. Accept terms and conditions")
    print("4. You may be asked to set up billing, but Gemini Flash has a generous free tier:")
    print("   - 15 requests per minute")
    print("   - 1 million tokens per minute")
    print("   - 1,500 requests per day")
    print("   - This should cover most of your Q&A generation needs!")

    input("\nPress Enter when you've completed Step 1...")

    print("\n" + "=" * 80)
    print("Step 2: Enable Generative Language API")
    print("=" * 80)
    print("\n1. Go to: https://aistudio.google.com/app/apikey")
    print("2. This is Google AI Studio - the easiest way to get a Gemini API key")
    print("3. OR alternatively:")
    print("   - Go to: https://console.cloud.google.com/apis/library")
    print("   - Search for 'Generative Language API'")
    print("   - Click 'Enable'")

    input("\nPress Enter when you've completed Step 2...")

    print("\n" + "=" * 80)
    print("Step 3: Create API Key")
    print("=" * 80)
    print("\nEASIEST METHOD - Google AI Studio:")
    print("1. Go to: https://aistudio.google.com/app/apikey")
    print("2. Click 'Create API Key'")
    print("3. Copy your API key (it starts with 'AIza...')")
    print("\nALTERNATIVE METHOD - Google Cloud Console:")
    print("1. Go to: https://console.cloud.google.com/apis/credentials")
    print("2. Click 'Create Credentials' → 'API Key'")
    print("3. Copy your API key (it starts with 'AIza...')")
    print("4. (Optional) Click 'Restrict Key' and limit to 'Generative Language API' only")

    print("\n" + "=" * 80)
    api_key = input("Paste your API key here: ").strip()

    if not api_key:
        print("\n❌ Error: No API key provided. Please run this script again.")
        return

    if not api_key.startswith('AIza'):
        print("\n⚠️  Warning: API key should start with 'AIza'. Please verify it's correct.")
        confirm = input("Continue anyway? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Setup cancelled. Please run this script again with the correct API key.")
            return

    # Save to .env file
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / '.env'

    with open(env_path, 'w') as f:
        f.write(f"GEMINI_API_KEY={api_key}\n")

    print(f"\n✅ API key saved to: {env_path}")
    print("\n⚠️  IMPORTANT SECURITY NOTES:")
    print("   1. Add '.env' to your .gitignore file to avoid committing your API key!")
    print("   2. Never share your API key publicly")
    print("   3. Keep your API key secure")

    # Check .gitignore
    gitignore_path = project_root / '.gitignore'
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            gitignore_content = f.read()
        if '.env' not in gitignore_content:
            print(f"\n⚠️  Warning: .env is not in .gitignore!")
            add_to_gitignore = input("   Add .env to .gitignore now? (y/n): ").strip().lower()
            if add_to_gitignore == 'y':
                with open(gitignore_path, 'a') as f:
                    f.write("\n# Environment variables\n.env\n")
                print("   ✅ Added .env to .gitignore")
    else:
        print(f"\n⚠️  No .gitignore found. Creating one...")
        with open(gitignore_path, 'w') as f:
            f.write("# Environment variables\n.env\n\n# Python\n__pycache__/\n*.pyc\n*.pyo\n\n# Data\ndata/\nlogs/\n")
        print("   ✅ Created .gitignore with .env entry")

    # Test API key
    print("\n" + "=" * 80)
    print("Step 4: Testing API Connection")
    print("=" * 80)

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        print("\nSending test request to Gemini API...")
        response = model.generate_content("Say 'Hello! I'm ready to generate agricultural Q&A pairs!'")

        print(f"\n✅ API test successful!")
        print(f"   Response from Gemini: {response.text}")

    except ImportError:
        print("\n❌ google-generativeai package not installed.")
        print("   Please run: pip install -r requirements.txt")
        print("   Then run this setup script again to test the API connection.")
        return
    except Exception as e:
        print(f"\n❌ API test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Verify your API key is correct")
        print("2. Make sure the Generative Language API is enabled in Google Cloud Console")
        print("3. Check if you have billing set up (required after free tier)")
        print("4. Try generating a new API key")
        return

    # Success summary
    print("\n" + "=" * 80)
    print("SETUP COMPLETE!")
    print("=" * 80)
    print("\n✅ Your Google Gemini API is configured and working!")
    print("\n📊 API Rate Limits (Free Tier):")
    print("   - 15 requests per minute")
    print("   - 1 million tokens per minute")
    print("   - 1,500 requests per day")
    print("\n💰 Pricing (after free tier):")
    print("   - Input: $0.075 per 1M tokens")
    print("   - Output: $0.30 per 1M tokens")
    print("   - Estimated project cost: ~$1-2 total for 50k Q&A pairs")
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("\n1. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n2. Create semantic chunks from your dataset:")
    print("   python qa_generation/scripts/02_create_chunks.py")
    print("\n3. Generate your first batch of Q&A pairs:")
    print("   python qa_generation/scripts/03_generate_batch.py --batch-id 1")
    print("\n" + "=" * 80)
    print("\nHappy Q&A generation! 🚀")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    setup_guide()
