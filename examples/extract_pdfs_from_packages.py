"""
Extract PDFs from tar.gz Packages

This script extracts PDF files from downloaded tar.gz packages from PMC OA Service
and updates the article registry with new PDF paths.

Usage:
    python examples/extract_pdfs_from_packages.py
"""

import sys
import os
import tarfile
import shutil
from pathlib import Path
from typing import Optional, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.article_registry import ArticleRegistry


# Configuration
PACKAGES_DIR = "articles/packages"
PDFS_DIR = "articles/pdfs"
TEMP_DIR = "articles/temp"
REGISTRY_PATH = "articles/metadata.json"


def print_separator(title: str = "", char: str = "="):
    """Print formatted separator"""
    if title:
        print(f"\n{char * 80}")
        print(f"  {title}")
        print(f"{char * 80}\n")
    else:
        print(f"{char * 80}\n")


def find_pdf_in_tar(tar_path: str, temp_dir: str) -> Optional[str]:
    """
    Extract tar.gz and find PDF file inside

    Args:
        tar_path: Path to tar.gz file
        temp_dir: Temporary directory for extraction

    Returns:
        Path to extracted PDF or None if not found
    """
    try:
        # Create temporary extraction directory
        extract_dir = Path(temp_dir) / Path(tar_path).stem
        extract_dir.mkdir(parents=True, exist_ok=True)

        # Extract tar.gz
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_dir)

        # Search for PDF files recursively
        pdf_files = list(extract_dir.rglob("*.pdf"))

        if not pdf_files:
            print(f"  ⚠ No PDF found in archive")
            return None

        if len(pdf_files) > 1:
            print(f"  ⚠ Multiple PDFs found ({len(pdf_files)}), using first one")

        return str(pdf_files[0])

    except Exception as e:
        print(f"  ✗ Error extracting archive: {e}")
        return None


def copy_pdf_to_output(pdf_path: str, output_dir: str, pmc_id: str) -> Optional[str]:
    """
    Copy PDF to output directory with PMC ID naming

    Args:
        pdf_path: Path to source PDF
        output_dir: Output directory
        pmc_id: PMC ID for naming

    Returns:
        Path to copied PDF or None on error
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create filename from PMC ID
        output_file = output_path / f"{pmc_id}.pdf"

        # Copy PDF
        shutil.copy2(pdf_path, output_file)

        return str(output_file)

    except Exception as e:
        print(f"  ✗ Error copying PDF: {e}")
        return None


def cleanup_temp_dir(temp_dir: str):
    """Remove temporary extraction directory"""
    try:
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"⚠ Warning: Could not remove temp directory: {e}")


def main():
    print_separator("Extract PDFs from tar.gz Packages")

    # Initialize registry
    registry = ArticleRegistry(registry_path=REGISTRY_PATH)

    print(f"Configuration:")
    print(f"  Packages directory: {PACKAGES_DIR}")
    print(f"  Output directory: {PDFS_DIR}")
    print(f"  Temporary directory: {TEMP_DIR}")
    print(f"  Registry: {REGISTRY_PATH}")
    print(f"\nRegistry status:")
    print(f"  Total articles: {registry.count()}")

    # Get all articles from registry
    all_articles = registry.list_all()

    # Filter articles that have tar.gz packages but no extracted PDF
    articles_to_process = []
    for article in all_articles:
        if article.pdf_path and article.pdf_path.endswith('.tar.gz'):
            # Check if PDF already extracted
            expected_pdf = Path(PDFS_DIR) / f"{article.pmc_id}.pdf"
            if not expected_pdf.exists():
                articles_to_process.append(article)

    if not articles_to_process:
        print(f"\n✓ No packages to extract (all PDFs already extracted)")
        return

    print(f"\nArticles to process: {len(articles_to_process)}")

    # Statistics
    extracted = 0
    already_exists = 0
    failed = 0

    # Create temp directory
    temp_path = Path(TEMP_DIR)
    temp_path.mkdir(parents=True, exist_ok=True)

    # Process each article
    print_separator("Extracting PDFs")

    for i, article in enumerate(articles_to_process, 1):
        print(f"\n[{i}/{len(articles_to_process)}] Processing {article.pmc_id}")
        print(f"  Title: {article.title}...")

        tar_path = article.pdf_path

        # Check if tar.gz exists
        if not Path(tar_path).exists():
            print(f"  ✗ Package not found: {tar_path}")
            failed += 1
            continue

        # Check if PDF already extracted
        output_pdf = Path(PDFS_DIR) / f"{article.pmc_id}.pdf"
        if output_pdf.exists():
            print(f"  ⚠ PDF already exists: {output_pdf}")
            already_exists += 1
            continue

        # Extract and find PDF
        print(f"  Extracting tar.gz...")
        pdf_path = find_pdf_in_tar(tar_path, TEMP_DIR)

        if not pdf_path:
            failed += 1
            continue

        print(f"  Found PDF: {Path(pdf_path).name}")

        # Copy PDF to output directory
        print(f"  Copying to output directory...")
        final_pdf_path = copy_pdf_to_output(pdf_path, PDFS_DIR, article.pmc_id)

        if not final_pdf_path:
            failed += 1
            continue

        # Update registry
        try:
            registry.update(article.pmid, pdf_path=final_pdf_path)
            file_size = os.path.getsize(final_pdf_path)
            print(f"  ✓ Extracted: {final_pdf_path} ({file_size / 1024:.1f} KB)")
            extracted += 1
        except Exception as e:
            print(f"  ✗ Error updating registry: {e}")
            failed += 1

        # Progress update every 10 articles
        if i % 10 == 0:
            print(f"\nProgress: {i}/{len(articles_to_process)} processed")
            print(f"  Extracted: {extracted} | Already exists: {already_exists} | Failed: {failed}")

    # Cleanup temporary directory
    print(f"\nCleaning up temporary files...")
    cleanup_temp_dir(TEMP_DIR)

    # Final summary
    print_separator("Extraction Complete")

    print(f"Summary:")
    print(f"  Total processed: {len(articles_to_process)}")
    print(f"  Successfully extracted: {extracted}")
    print(f"  Already existed: {already_exists}")
    print(f"  Failed: {failed}")

    if extracted > 0:
        # Calculate total size of extracted PDFs
        pdf_dir = Path(PDFS_DIR)
        if pdf_dir.exists():
            pdf_files = list(pdf_dir.glob("*.pdf"))
            total_size = sum(f.stat().st_size for f in pdf_files)
            print(f"\nPDF Statistics:")
            print(f"  Total PDFs: {len(pdf_files)}")
            print(f"  Total size: {total_size / (1024 * 1024):.2f} MB")
            print(f"  Average size: {total_size / len(pdf_files) / 1024:.1f} KB")

    # Registry statistics
    print_separator("Updated Registry Statistics")
    stats = registry.get_stats()
    print(f"Total articles in registry: {stats['total_articles']}")
    print(f"Total size: {stats['total_size_mb']} MB")

    # Count PDFs vs tar.gz
    all_articles_updated = registry.list_all()
    pdf_count = sum(1 for a in all_articles_updated if a.pdf_path and a.pdf_path.endswith('.pdf'))
    tar_count = sum(1 for a in all_articles_updated if a.pdf_path and a.pdf_path.endswith('.tar.gz'))

    print(f"\nFiles in registry:")
    print(f"  PDFs extracted: {pdf_count}")
    print(f"  tar.gz packages only: {tar_count}")

    print_separator()
    print(f"\n✓ All PDFs saved to: {PDFS_DIR}")
    print(f"✓ Registry updated: {REGISTRY_PATH}")

    if tar_count > 0:
        print(f"\nNote: {tar_count} packages still need extraction (run script again if needed)")


if __name__ == "__main__":
    main()
