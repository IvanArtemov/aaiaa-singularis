"""
Launch SciBERT-Nebius Telegram Bot for PDF to Knowledge Graph conversion

This script starts the Telegram bot that:
1. Receives PDF files from users
2. Extracts entities and relationships using SciBERT-Nebius pipeline
3. Generates SVG knowledge graph
4. Sends results back to user

Key Features:
- FREE SciBERT embeddings (domain-optimized for scientific papers)
- Nebius gpt-oss-120b LLM (cost-efficient)
- GROBID ML parser (structured IMRAD extraction)
- ChromaDB semantic search
- Parallel entity validation

Requirements:
- TELEGRAM_BOT_TOKEN in .env
- NEBIUS_API_KEY in .env (get from: https://studio.nebius.com/)

Cost: ~$0.018 per paper (40% cheaper than LLM bot)

Usage:
    python scripts/run_scibert_telegram_bot.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.scibert_telegram_bot import main


if __name__ == "__main__":
    print("=" * 80)
    print("SCIBERT-NEBIUS TELEGRAM BOT - PDF to Knowledge Graph")
    print("=" * 80)
    print()
    print("🤖 Starting SciBERT-Nebius Telegram Bot...")
    print()
    print("Configuration:")
    print(f"  • Project root: {project_root}")
    print(f"  • Temp directory: {project_root / 'bot' / 'temp'}")
    print()
    print("📝 Make sure you have set in .env:")
    print("  • TELEGRAM_BOT_TOKEN - from @BotFather")
    print("  • NEBIUS_API_KEY - from https://studio.nebius.com/")
    print()
    print("✨ Pipeline Features:")
    print("  • SciBERT embeddings (FREE, domain-optimized)")
    print("  • Nebius gpt-oss-120b LLM (cost-efficient)")
    print("  • GROBID parser (ML-based IMRAD extraction)")
    print("  • ChromaDB semantic search")
    print("  • Parallel entity validation (4 threads)")
    print()
    print("💰 Cost: ~$0.018 per paper")
    print("⏱ Processing time: 60-90 seconds")
    print()
    print("🚀 Bot is starting...")
    print("   Press Ctrl+C to stop")
    print()
    print("=" * 80)
    print()

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("Bot stopped by user")
        print("=" * 80)
    except Exception as e:
        print("\n\n" + "=" * 80)
        print(f"❌ Error: {e}")
        print("=" * 80)
        raise
