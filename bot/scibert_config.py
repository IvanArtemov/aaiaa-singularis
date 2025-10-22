"""Configuration for SciBERT-Nebius Telegram Bot"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SciBertBotConfig:
    """SciBERT-Nebius Telegram Bot configuration"""

    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

    # Nebius (for SciBertNebiusPipeline)
    NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")

    # File handling
    PROJECT_ROOT = Path(__file__).parent.parent
    TEMP_DIR = PROJECT_ROOT / "bot" / "temp"
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_PDF_SIZE_MB", "10"))
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

    # Rate limiting
    MAX_REQUESTS_PER_USER_PER_HOUR = int(os.getenv("MAX_REQUESTS_PER_USER_PER_HOUR", "5"))

    # Processing
    PROCESSING_TIMEOUT_SECONDS = int(os.getenv("PROCESSING_TIMEOUT_SECONDS", "300"))  # 5 minutes for SciBERT

    # Cleanup
    CLEANUP_TEMP_FILES_OLDER_THAN_HOURS = 1

    # Results storage
    KEEP_PROCESSED_FILES = os.getenv("KEEP_PROCESSED_FILES", "true").lower() == "true"
    RESULTS_DIR = PROJECT_ROOT / "bot" / "results"
    ORGANIZE_BY_USER = True  # Create user_{user_id} subdirectories

    # Messages
    WELCOME_MESSAGE = """👋 Привет! Я бот для анализа научных статей (SciBERT + Nebius).

📄 Отправь мне PDF научной статьи, и я:
✓ Извлеку ключевые сущности (факты, гипотезы, результаты)
✓ Построю граф знаний
✓ Создам красивую SVG визуализацию

⚡ Стоимость: ~$0.018 за статью (дешевле!)
⏱ Время обработки: 60-90 секунд
📊 Ограничение: {max_requests} запросов в час

✨ Использует:
• SciBERT embeddings (FREE, domain-optimized)
• Nebius gpt-oss-120b (cost-efficient LLM)
• GROBID parser (ML-based extraction)

Попробуй! Просто отправь PDF файл."""

    HELP_MESSAGE = """📖 Инструкция по использованию

1️⃣ Отправь PDF файл научной статьи
2️⃣ Подожди 60-90 секунд обработки
3️⃣ Получи SVG граф знаний

📏 Ограничения:
• Максимальный размер файла: {max_size} MB
• Макс. запросов в час: {max_requests}

🔧 Команды:
/start - начать работу
/help - эта справка
/stats - статистика

💡 Технологии:
• SciBERT embeddings (FREE)
• Nebius gpt-oss-120b LLM
• ChromaDB semantic search
• GROBID structured parser

❓ Вопросы? Напиши @your_support"""

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        if not cls.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN not set in environment")
        if not cls.NEBIUS_API_KEY:
            raise ValueError("NEBIUS_API_KEY not set in environment. Get it from: https://studio.nebius.com/")

        # Ensure temp directory exists
        cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)

        # Ensure results directory exists if keeping processed files
        if cls.KEEP_PROCESSED_FILES:
            cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        return True

    @classmethod
    def get_welcome_message(cls) -> str:
        """Get formatted welcome message"""
        return cls.WELCOME_MESSAGE.format(
            max_requests=cls.MAX_REQUESTS_PER_USER_PER_HOUR
        )

    @classmethod
    def get_help_message(cls) -> str:
        """Get formatted help message"""
        return cls.HELP_MESSAGE.format(
            max_size=cls.MAX_FILE_SIZE_MB,
            max_requests=cls.MAX_REQUESTS_PER_USER_PER_HOUR
        )
