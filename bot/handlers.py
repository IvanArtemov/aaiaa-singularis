"""Telegram bot handlers for PDF processing"""

import os
import logging
from pathlib import Path
from telegram import Update
from telegram.ext import ContextTypes

from .config import BotConfig
from .session_manager import SessionManager
from .utils import (
    validate_pdf_file,
    generate_unique_filename,
    safe_delete_file,
    format_metrics,
    get_processing_status_message,
    create_error_message,
    save_result_to_temp
)
from .exceptions import (
    FileTooLargeError,
    InvalidFileTypeError,
    PDFParsingError,
    ExtractionError,
    SVGGenerationError,
    RateLimitExceededError
)

from src.parsers import PDFParser
from src.pipelines import LLMPipeline
from src.visualization.generate_svg import generate_svg_from_json

logger = logging.getLogger(__name__)


class BotHandlers:
    """Telegram bot message handlers"""

    def __init__(self, session_manager: SessionManager, config: BotConfig):
        self.session_manager = session_manager
        self.config = config
        self.pdf_parser = PDFParser()

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user
        logger.info(f"User {user.id} ({user.username}) started bot")

        await update.message.reply_text(
            self.config.get_welcome_message(),
            parse_mode='HTML'
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        await update.message.reply_text(
            self.config.get_help_message(),
            parse_mode='HTML'
        )

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command"""
        user = update.effective_user
        stats = self.session_manager.get_user_stats(user.id)

        message = f"""📊 Ваша статистика

📈 Всего запросов: {stats['total_requests']}
💰 Суммарная стоимость: ${stats['total_cost']:.4f}
📦 Извлечено сущностей: {stats['total_entities']}
⏱ Запросов за последний час: {stats['requests_last_hour']}/{self.config.MAX_REQUESTS_PER_USER_PER_HOUR}"""

        if stats['first_request']:
            message += f"\n🕒 Первый запрос: {stats['first_request'][:10]}"

        await update.message.reply_text(message)

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle document uploads (PDF files)"""
        user = update.effective_user
        document = update.message.document

        logger.info(f"User {user.id} uploaded document: {document.file_name}")

        # Validate file
        is_valid, error_msg = validate_pdf_file(
            document.file_name,
            document.file_size,
            self.config.MAX_FILE_SIZE_BYTES
        )

        if not is_valid:
            await update.message.reply_text(error_msg)
            return

        try:
            # Check rate limit
            self.session_manager.check_rate_limit(user.id)

            # Check if already processing
            if self.session_manager.is_processing(user.id):
                await update.message.reply_text(
                    "⏳ У вас уже обрабатывается файл. Пожалуйста, дождитесь завершения."
                )
                return

            # Mark as processing
            self.session_manager.start_processing(user.id)

            # Process the PDF
            await self._process_pdf(update, context, document)

        except RateLimitExceededError as e:
            await update.message.reply_text(create_error_message(e))

        except Exception as e:
            logger.error(f"Error processing document: {e}", exc_info=True)
            await update.message.reply_text(create_error_message(e))

        finally:
            # Mark as no longer processing
            self.session_manager.finish_processing(user.id)

    async def _process_pdf(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        document
    ):
        """
        Process PDF through full pipeline

        Args:
            update: Telegram update
            context: Telegram context
            document: Telegram document object
        """
        user = update.effective_user
        temp_paths = []  # Track files to cleanup

        try:
            # Step 1: Download PDF
            status_msg = await update.message.reply_text(
                get_processing_status_message("downloading")
            )

            file = await document.get_file()
            unique_filename = generate_unique_filename(user.id, document.file_name)
            pdf_path = self.config.TEMP_DIR / unique_filename
            temp_paths.append(pdf_path)

            await file.download_to_drive(pdf_path)
            logger.info(f"Downloaded PDF to {pdf_path}")

            # Step 2: Parse PDF
            await status_msg.edit_text(
                get_processing_status_message("parsing")
            )

            try:
                parsed_doc = self.pdf_parser.parse(str(pdf_path))
                logger.info(f"Parsed PDF: {parsed_doc.page_count} pages, {parsed_doc.word_count} words")
            except Exception as e:
                raise PDFParsingError(e)

            await status_msg.edit_text(
                get_processing_status_message(
                    "parsing_done",
                    pages=parsed_doc.page_count,
                    words=parsed_doc.word_count,
                    sections=len(parsed_doc.sections)
                )
            )

            # Step 3: Extract entities with LLM
            await status_msg.edit_text(
                get_processing_status_message("extracting")
            )

            try:
                pipeline = LLMPipeline(
                    config=self.config.LLM_CONFIG,
                    api_key=self.config.OPENAI_API_KEY,
                    model=self.config.LLM_MODEL
                )

                paper_id = f"tg_{user.id}_{unique_filename.replace('.pdf', '')}"

                result = pipeline.extract(
                    paper_text=parsed_doc.text,
                    paper_id=paper_id,
                    metadata=parsed_doc.metadata
                )

                logger.info(
                    f"Extraction complete: {result.total_entities()} entities, "
                    f"{result.total_relationships()} relationships, "
                    f"cost: ${result.metrics.cost_usd:.4f}"
                )

            except Exception as e:
                raise ExtractionError(e)

            # Step 4: Generate SVG
            await status_msg.edit_text(
                get_processing_status_message("generating_svg")
            )

            try:
                # Save result to temp JSON
                result_json_path = save_result_to_temp(result, self.config.TEMP_DIR, user.id)
                temp_paths.append(result_json_path)

                # Generate SVG
                svg_filename = f"graph_{user.id}_{unique_filename.replace('.pdf', '.svg')}"
                svg_path = self.config.TEMP_DIR / svg_filename
                temp_paths.append(svg_path)

                generate_svg_from_json(str(result_json_path), str(svg_path))
                logger.info(f"Generated SVG: {svg_path}")

            except Exception as e:
                raise SVGGenerationError(e)

            # Step 5: Send results
            await status_msg.edit_text(
                get_processing_status_message("complete")
            )

            # Send SVG file
            with open(svg_path, 'rb') as svg_file:
                await context.bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=svg_file,
                    filename=f"knowledge_graph.svg",
                    caption=format_metrics(result)
                )

            # Update session stats
            self.session_manager.finish_processing(
                user.id,
                cost=result.metrics.cost_usd,
                entities=result.total_entities()
            )

            logger.info(f"Successfully processed PDF for user {user.id}")

        except (PDFParsingError, ExtractionError, SVGGenerationError) as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            await update.message.reply_text(create_error_message(e))
            raise

        finally:
            # Cleanup temporary files
            for path in temp_paths:
                safe_delete_file(path)

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages"""
        await update.message.reply_text(
            "📄 Отправьте мне PDF файл научной статьи для анализа.\n\n"
            "Используйте /help для получения инструкций."
        )

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        logger.error(f"Update {update} caused error {context.error}", exc_info=context.error)

        if update and update.effective_message:
            await update.effective_message.reply_text(
                "❌ Произошла непредвиденная ошибка. Пожалуйста, попробуйте позже."
            )
