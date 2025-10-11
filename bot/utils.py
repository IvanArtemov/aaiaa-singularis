"""Utility functions for Telegram Bot"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta

from src.models import ExtractionResult


def safe_delete_file(file_path: str | Path) -> bool:
    """
    Safely delete a file

    Args:
        file_path: Path to file to delete

    Returns:
        True if deleted, False otherwise
    """
    try:
        file_path = Path(file_path)
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    except Exception as e:
        print(f"Warning: Failed to delete {file_path}: {e}")
        return False


def cleanup_temp_files(temp_dir: Path, max_age_hours: int = 1):
    """
    Clean up temporary files older than max_age_hours

    Args:
        temp_dir: Directory containing temp files
        max_age_hours: Maximum age in hours before cleanup
    """
    if not temp_dir.exists():
        return

    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

    for file_path in temp_dir.iterdir():
        if not file_path.is_file():
            continue

        # Check file modification time
        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

        if file_mtime < cutoff_time:
            safe_delete_file(file_path)


def generate_unique_filename(user_id: int, original_filename: str) -> str:
    """
    Generate unique filename for user's file

    Args:
        user_id: Telegram user ID
        original_filename: Original filename

    Returns:
        Unique filename with timestamp
    """
    timestamp = int(time.time())
    extension = Path(original_filename).suffix
    return f"user_{user_id}_{timestamp}{extension}"


def format_metrics(result: ExtractionResult) -> str:
    """
    Format extraction result metrics for user

    Args:
        result: ExtractionResult from pipeline

    Returns:
        Formatted string for Telegram message
    """
    # Entity counts
    entity_lines = []
    for entity_type, entities in result.entities.items():
        if entities:
            entity_lines.append(f"   {entity_type.capitalize()}: {len(entities)}")

    entity_counts = "\n".join(entity_lines) if entity_lines else "   None extracted"

    message = f"""✅ Extraction Complete!

📊 Statistics:
• Entities: {result.total_entities()}
• Relationships: {result.total_relationships()}
• Processing time: {result.metrics.processing_time:.1f}s
• Cost: ${result.metrics.cost_usd:.4f}

📦 Entities by type:
{entity_counts}

💡 Open the SVG file to explore the knowledge graph!"""

    return message


def validate_pdf_file(file_name: str, file_size: int, max_size_bytes: int) -> tuple[bool, Optional[str]]:
    """
    Validate PDF file

    Args:
        file_name: Name of the file
        file_size: Size of file in bytes
        max_size_bytes: Maximum allowed size in bytes

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file extension
    if not file_name.lower().endswith('.pdf'):
        return False, "❌ Пожалуйста, отправьте PDF файл"

    # Check file size
    if file_size > max_size_bytes:
        max_size_mb = max_size_bytes / (1024 * 1024)
        actual_size_mb = file_size / (1024 * 1024)
        return False, f"❌ Файл слишком большой ({actual_size_mb:.1f} MB). Максимум: {max_size_mb:.0f} MB"

    return True, None


def format_entity_preview(entity: dict, max_length: int = 60) -> str:
    """
    Format entity for preview

    Args:
        entity: Entity dict with 'text' field
        max_length: Maximum length of preview

    Returns:
        Formatted string
    """
    text = entity.get('text', '')
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "2.5 MB")
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def save_result_to_temp(result: ExtractionResult, temp_dir: Path, user_id: int) -> Path:
    """
    Save extraction result to temporary JSON file

    Args:
        result: ExtractionResult to save
        temp_dir: Temporary directory
        user_id: User ID

    Returns:
        Path to saved JSON file
    """
    timestamp = int(time.time())
    filename = f"result_user_{user_id}_{timestamp}.json"
    file_path = temp_dir / filename

    result.to_json(str(file_path))

    return file_path


def get_processing_status_message(step: str, **kwargs) -> str:
    """
    Get formatted status message for different processing steps

    Args:
        step: Processing step name
        **kwargs: Additional parameters for formatting

    Returns:
        Formatted status message
    """
    messages = {
        "downloading": "📥 Downloading PDF...",
        "parsing": "📄 Parsing PDF...",
        "parsing_done": """📄 Parsing PDF... ✅
   • Pages: {pages}
   • Words: {words}
   • Sections: {sections}""",
        "extracting": """🤖 Extracting knowledge...
   This will take 30-60 seconds ⏳""",
        "generating_svg": "🎨 Generating knowledge graph...",
        "complete": "✨ Processing complete!"
    }

    template = messages.get(step, step)
    return template.format(**kwargs)


def create_error_message(error: Exception) -> str:
    """
    Create user-friendly error message

    Args:
        error: Exception that occurred

    Returns:
        Formatted error message
    """
    from .exceptions import (
        FileTooLargeError,
        InvalidFileTypeError,
        PDFParsingError,
        ExtractionError,
        SVGGenerationError,
        RateLimitExceededError,
        ProcessingTimeoutError
    )

    if isinstance(error, FileTooLargeError):
        return f"❌ {str(error)}"

    elif isinstance(error, InvalidFileTypeError):
        return "❌ Пожалуйста, отправьте PDF файл"

    elif isinstance(error, PDFParsingError):
        return """❌ Не удалось обработать PDF

Возможные причины:
• Файл поврежден
• PDF зашифрован/защищен
• Неподдерживаемый формат

Попробуйте другой файл."""

    elif isinstance(error, ExtractionError):
        return """❌ Ошибка при извлечении данных

Попробуйте:
• Отправить другой файл
• Попробовать позже
• Обратиться в поддержку"""

    elif isinstance(error, SVGGenerationError):
        return """❌ Ошибка при создании графа

Данные извлечены, но не удалось создать визуализацию.
Попробуйте еще раз или обратитесь в поддержку."""

    elif isinstance(error, RateLimitExceededError):
        return f"""⏸ Превышен лимит запросов

Вы можете отправить максимум {error.limit} запросов в час.
Попробуйте позже."""

    elif isinstance(error, ProcessingTimeoutError):
        return """⏱ Превышено время обработки

Файл слишком большой или сложный.
Попробуйте отправить статью меньшего размера."""

    else:
        return f"""❌ Произошла ошибка

Попробуйте еще раз или обратитесь в поддержку.

Техническая информация: {type(error).__name__}"""
