"""Configuration for Telegram Bot"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BotConfig:
    """Telegram Bot configuration"""

    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

    # OpenAI (for LLM pipeline)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # File handling
    PROJECT_ROOT = Path(__file__).parent.parent
    TEMP_DIR = PROJECT_ROOT / "bot" / "temp"
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_PDF_SIZE_MB", "10"))
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

    # Rate limiting
    MAX_REQUESTS_PER_USER_PER_HOUR = int(os.getenv("MAX_REQUESTS_PER_USER_PER_HOUR", "5"))

    # Processing
    PROCESSING_TIMEOUT_SECONDS = int(os.getenv("PROCESSING_TIMEOUT_SECONDS", "180"))  # 3 minutes

    # LLM Pipeline config
    LLM_CONFIG = {
        "temperature": 0.1,
        "max_tokens": 4000
    }
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5-mini")

    # Cleanup
    CLEANUP_TEMP_FILES_OLDER_THAN_HOURS = 1

    # Messages
    WELCOME_MESSAGE = """👋 Привет! Я бот для анализа научных статей.

📄 Отправь мне PDF научной статьи, и я:
✓ Извлеку ключевые сущности (факты, гипотезы, результаты)
✓ Построю граф знаний
✓ Создам красивую SVG визуализацию

⚡ Стоимость: ~$0.03 за статью
⏱ Время обработки: 30-60 секунд
📊 Ограничение: {max_requests} запросов в час

Попробуй! Просто отправь PDF файл."""

    HELP_MESSAGE = """📖 Инструкция по использованию

1️⃣ Отправь PDF файл научной статьи
2️⃣ Подожди 30-60 секунд обработки
3️⃣ Получи SVG граф знаний

📏 Ограничения:
• Максимальный размер файла: {max_size} MB
• Макс. запросов в час: {max_requests}

🔧 Команды:
/start - начать работу
/help - эта справка
/stats - статистика

❓ Вопросы? Напиши @your_support"""

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        if not cls.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN not set in environment")
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment")

        # Ensure temp directory exists
        cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)

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
