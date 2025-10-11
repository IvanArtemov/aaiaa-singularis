"""
Launch Telegram Bot for PDF to Knowledge Graph conversion

This script starts the Telegram bot that:
1. Receives PDF files from users
2. Extracts entities and relationships using LLM
3. Generates SVG knowledge graph
4. Sends results back to user

Requirements:
- TELEGRAM_BOT_TOKEN in .env
- OPENAI_API_KEY in .env

Usage:
    python examples/run_telegram_bot.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.telegram_bot import main


if __name__ == "__main__":
    print("=" * 80)
    print("TELEGRAM BOT - PDF to Knowledge Graph")
    print("=" * 80)
    print()
    print("🤖 Starting Telegram Bot...")
    print()
    print("Configuration:")
    print(f"  • Project root: {project_root}")
    print(f"  • Temp directory: {project_root / 'bot' / 'temp'}")
    print()
    print("📝 Make sure you have set in .env:")
    print("  • TELEGRAM_BOT_TOKEN - from @BotFather")
    print("  • OPENAI_API_KEY - your OpenAI API key")
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
