"""Main Telegram Bot application"""

import logging
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters
)

from .config import BotConfig
from .session_manager import SessionManager
from .handlers import BotHandlers
from .utils import cleanup_temp_files

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram Bot for PDF to Knowledge Graph conversion"""

    def __init__(self):
        """Initialize bot"""
        # Validate configuration
        BotConfig.validate()

        # Initialize components
        self.config = BotConfig
        self.session_manager = SessionManager(
            max_requests_per_hour=self.config.MAX_REQUESTS_PER_USER_PER_HOUR
        )
        self.handlers = BotHandlers(self.session_manager, self.config)

        # Create application
        self.application = Application.builder().token(self.config.TELEGRAM_BOT_TOKEN).build()

        # Register handlers
        self._register_handlers()

        logger.info("Telegram Bot initialized successfully")

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
        logger.info("Bot started successfully!")
        logger.info(f"Temp directory: {self.config.TEMP_DIR}")
        logger.info(f"Max file size: {self.config.MAX_FILE_SIZE_MB} MB")
        logger.info(f"Rate limit: {self.config.MAX_REQUESTS_PER_USER_PER_HOUR} requests/hour")

    async def post_shutdown(self, application: Application):
        """Cleanup tasks before shutdown"""
        logger.info("Shutting down bot...")

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
        logger.info("Starting Telegram Bot...")

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
        bot = TelegramBot()
        bot.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
