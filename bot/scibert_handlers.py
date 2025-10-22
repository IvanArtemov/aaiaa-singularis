"""Telegram bot handlers for PDF processing with SciBERT-Nebius pipeline"""

import logging
from pathlib import Path
from telegram import Update
from telegram.ext import ContextTypes

from .scibert_config import SciBertBotConfig
from .session_manager import SessionManager
from .utils import (
    validate_pdf_file,
    generate_unique_filename,
    safe_delete_file,
    format_metrics,
    get_processing_status_message,
    create_error_message,
    save_result_to_temp,
    create_article_directory,
    save_article_metadata,
    save_parsed_document
)
from .exceptions import (
    FileTooLargeError,
    InvalidFileTypeError,
    PDFParsingError,
    ExtractionError,
    SVGGenerationError,
    RateLimitExceededError
)

from src.parsers import GrobidParser  # Use GROBID for IMRAD sections
from src.pipelines import SciBertNebiusPipeline
from src.visualization.generate_svg import generate_svg_from_json

logger = logging.getLogger(__name__)


class SciBertBotHandlers:
    """Telegram bot message handlers using SciBERT-Nebius pipeline"""

    def __init__(self, session_manager: SessionManager, config: SciBertBotConfig):
        self.session_manager = session_manager
        self.config = config
        self.pdf_parser = GrobidParser()  # Use GROBID instead of PyMuPDF

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user
        logger.info(f"User {user.id} ({user.username}) started SciBERT bot")

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

        message = f"""📊 Ваша статистика (SciBERT-Nebius Bot)

📈 Всего запросов: {stats['total_requests']}
💰 Суммарная стоимость: ${stats['total_cost']:.4f}
📦 Извлечено сущностей: {stats['total_entities']}
⏱ Запросов за последний час: {stats['requests_last_hour']}/{self.config.MAX_REQUESTS_PER_USER_PER_HOUR}"""

        if stats['first_request']:
            message += f"\n🕒 Первый запрос: {stats['first_request'][:10]}"

        message += f"\n\n✨ Pipeline: SciBERT + Nebius gpt-oss-120b"

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
            # self.session_manager.check_rate_limit(user.id)

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
        Process PDF through SciBERT-Nebius pipeline

        Args:
            update: Telegram update
            context: Telegram context
            document: Telegram document object
        """
        user = update.effective_user
        temp_paths = []  # Track files to cleanup
        article_dir = None  # Will hold article directory if KEEP_PROCESSED_FILES is True

        try:
            # Determine where to save files
            if self.config.KEEP_PROCESSED_FILES:
                # Create dedicated article directory
                article_dir = create_article_directory(
                    self.config.RESULTS_DIR,
                    user.id,
                    document.file_name
                )
                logger.info(f"Created article directory: {article_dir}")

            # Step 1: Download PDF
            status_msg = await update.message.reply_text(
                get_processing_status_message("downloading")
            )

            file = await document.get_file()
            unique_filename = generate_unique_filename(user.id, document.file_name)

            # Save to article directory or temp
            if article_dir:
                pdf_path = article_dir / document.file_name
            else:
                pdf_path = self.config.TEMP_DIR / unique_filename
                temp_paths.append(pdf_path)

            await file.download_to_drive(pdf_path)
            logger.info(f"Downloaded PDF to {pdf_path}")

            # Step 2: Parse PDF with GROBID
            await status_msg.edit_text(
                get_processing_status_message("parsing") + "\n(Using GROBID ML parser...)"
            )

            try:
                parsed_doc = self.pdf_parser.parse(str(pdf_path))
                logger.info(
                    f"Parsed PDF with GROBID: {parsed_doc.word_count} words, "
                    f"IMRAD sections: {list(parsed_doc.imrad_sections.keys()) if parsed_doc.imrad_sections else 'none'}"
                )
            except Exception as e:
                raise PDFParsingError(e)

            # Update status with IMRAD sections info
            sections_info = ""
            if parsed_doc.imrad_sections:
                sections_info = f"\nIMRAD sections: {', '.join(parsed_doc.imrad_sections.keys())}"

            await status_msg.edit_text(
                f"✅ PDF parsed: {parsed_doc.word_count} words{sections_info}"
            )

            # Save parsed document if keeping files
            if article_dir:
                save_parsed_document(article_dir, parsed_doc)
                logger.info(f"Saved parsed document to {article_dir / 'parsed_doc.json'}")

            # Step 3: Extract entities with SciBERT-Nebius pipeline
            await status_msg.edit_text(
                "🔬 Extracting entities with SciBERT-Nebius pipeline...\n"
                "Phase 1: SciBERT embeddings (FREE)\n"
                "Phase 2: Keyword generation (Nebius)\n"
                "Phase 3: Semantic retrieval (ChromaDB)\n"
                "Phase 4: Entity validation (Nebius)"
            )

            try:
                # Initialize pipeline (config from scibert_nebius_config.yaml)
                pipeline = SciBertNebiusPipeline()

                paper_id = f"tg_scibert_{user.id}_{unique_filename.replace('.pdf', '')}"

                result = pipeline.extract(
                    parsed_doc=parsed_doc,
                    paper_id=paper_id
                )

                logger.info(
                    f"SciBERT-Nebius extraction complete: {result.total_entities()} entities, "
                    f"{result.total_relationships()} relationships, "
                    f"cost: ${result.metrics.cost_usd:.4f}"
                )

            except Exception as e:
                logger.error(f"Extraction error: {e}", exc_info=True)
                raise ExtractionError(e)

            # Step 4: Generate SVG
            await status_msg.edit_text(
                "🎨 Generating knowledge graph visualization..."
            )

            try:
                # Save result JSON
                if article_dir:
                    result_json_path = article_dir / "extraction_result.json"
                    result.to_json(str(result_json_path))
                else:
                    result_json_path = save_result_to_temp(result, self.config.TEMP_DIR, user.id)
                    temp_paths.append(result_json_path)

                logger.info(f"Saved extraction result to {result_json_path}")

                # Generate SVG
                if article_dir:
                    svg_path = article_dir / "knowledge_graph.svg"
                else:
                    svg_filename = f"graph_scibert_{user.id}_{unique_filename.replace('.pdf', '.svg')}"
                    svg_path = self.config.TEMP_DIR / svg_filename
                    temp_paths.append(svg_path)

                generate_svg_from_json(str(result_json_path), str(svg_path))
                logger.info(f"Generated SVG: {svg_path}")

                # Save metadata if keeping files
                if article_dir:
                    save_article_metadata(
                        article_dir,
                        user.id,
                        document.file_name,
                        "scibert_nebius",
                        result
                    )
                    logger.info(f"Saved metadata to {article_dir / 'metadata.json'}")

            except Exception as e:
                raise SVGGenerationError(e)

            # Step 5: Send results
            await status_msg.edit_text(
                "✅ Processing complete! Sending results..."
            )

            # Format caption with SciBERT-specific info
            caption = format_metrics(result)
            caption += "\n\n✨ Pipeline: SciBERT + Nebius gpt-oss-120b"

            # Send SVG file
            with open(svg_path, 'rb') as svg_file:
                await context.bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=svg_file,
                    filename=f"knowledge_graph_scibert.svg",
                    caption=caption
                )

            # Update session stats
            self.session_manager.finish_processing(
                user.id,
                cost=result.metrics.cost_usd,
                entities=result.total_entities()
            )

            logger.info(f"Successfully processed PDF for user {user.id} with SciBERT-Nebius pipeline")

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
            "✨ Использую SciBERT embeddings + Nebius LLM\n"
            "💰 Стоимость: ~$0.018/статья\n\n"
            "Используйте /help для получения инструкций."
        )

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        logger.error(f"Update {update} caused error {context.error}", exc_info=context.error)

        if update and update.effective_message:
            await update.effective_message.reply_text(
                "❌ Произошла непредвиденная ошибка. Пожалуйста, попробуйте позже."
            )
