"""SciBERT-Nebius Telegram Bot application"""

import logging
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters
)

from .scibert_config import SciBertBotConfig
from .session_manager import SessionManager
from .scibert_handlers import SciBertBotHandlers
from .utils import cleanup_temp_files

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class SciBertTelegramBot:
    """Telegram Bot for PDF to Knowledge Graph conversion using SciBERT-Nebius pipeline"""

    def __init__(self):
        """Initialize bot"""
        # Validate configuration
        SciBertBotConfig.validate()

        # Initialize components
        self.config = SciBertBotConfig
        self.session_manager = SessionManager(
            max_requests_per_hour=self.config.MAX_REQUESTS_PER_USER_PER_HOUR
        )
        self.handlers = SciBertBotHandlers(self.session_manager, self.config)

        # Create application
        self.application = Application.builder().token(self.config.TELEGRAM_BOT_TOKEN).build()

        # Register handlers
        self._register_handlers()

        logger.info("SciBERT-Nebius Telegram Bot initialized successfully")

    def _register_handlers(self):
        """Register all command and message handlers"""
        # Command handlers
        self.application.add_handler(
            CommandHandler("start", self.handlers.start_command)
        )
        self.application.add_handler(
            CommandHandler("help", self.handlers.help_command)
        )
        self.application.add_handler(
            CommandHandler("stats", self.handlers.stats_command)
        )

        # Document handler (for PDFs)
        self.application.add_handler(
            MessageHandler(
                filters.Document.PDF | filters.Document.ALL,
                self.handlers.handle_document
            )
        )

        # Text message handler (fallback)
        self.application.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self.handlers.handle_text
            )
        )

        # Error handler
        self.application.add_error_handler(self.handlers.error_handler)

        logger.info("Handlers registered")

    async def post_init(self, application: Application):
        """Post-initialization tasks"""
        logger.info("SciBERT-Nebius Bot started successfully!")
        logger.info(f"Temp directory: {self.config.TEMP_DIR}")
        logger.info(f"Max file size: {self.config.MAX_FILE_SIZE_MB} MB")
        logger.info(f"Rate limit: {self.config.MAX_REQUESTS_PER_USER_PER_HOUR} requests/hour")
        logger.info("Pipeline: SciBERT embeddings + Nebius gpt-oss-120b LLM")

    async def post_shutdown(self, application: Application):
        """Cleanup tasks before shutdown"""
        logger.info("Shutting down SciBERT-Nebius bot...")

        # Cleanup old temp files
        cleanup_temp_files(
            self.config.TEMP_DIR,
            max_age_hours=self.config.CLEANUP_TEMP_FILES_OLDER_THAN_HOURS
        )

        # Cleanup old sessions
        self.session_manager.cleanup_old_sessions(max_age_days=7)

        logger.info("Cleanup complete")

    def run(self):
        """Start the bot"""
        logger.info("Starting SciBERT-Nebius Telegram Bot...")

        # Add post-init and post-shutdown hooks
        self.application.post_init = self.post_init
        self.application.post_shutdown = self.post_shutdown

        # Start polling
        self.application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )


def main():
    """Main entry point"""
    try:
        bot = SciBertTelegramBot()
        bot.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
