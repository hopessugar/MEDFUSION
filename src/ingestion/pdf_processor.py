"""PDF document processor for MedFusion-RAFT.

Extracts text from PDF files with structure preservation, page tracking,
and OCR fallback for scanned documents.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from src.models.schemas import Document, DocumentPage

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    logger.warning("PyMuPDF not available. PDF extraction will be limited.")
    PYMUPDF_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    logger.warning("OCR dependencies (pdf2image, pytesseract) not available. Scanned PDFs will not be processed.")
    OCR_AVAILABLE = False


class PDFProcessorConfig(BaseModel):
    """Configuration for PDF processing."""
    
    extract_images: bool = Field(
        default=False,
        description="Whether to extract embedded images from PDF"
    )
    
    ocr_fallback: bool = Field(
        default=True,
        description="Use OCR for pages with minimal text extraction"
    )
    
    ocr_threshold: int = Field(
        default=50,
        description="Minimum characters before triggering OCR fallback"
    )
    
    preserve_layout: bool = Field(
        default=True,
        description="Attempt to preserve document layout structure"
    )
    
    extract_metadata: bool = Field(
        default=True,
        description="Extract PDF metadata (author, creation date, etc.)"
    )
    
    max_pages: Optional[int] = Field(
        default=None,
        description="Maximum pages to process (None = all pages)"
    )
    
    dpi: int = Field(
        default=300,
        description="DPI for OCR image conversion"
    )


class PDFProcessor:
    """Process PDF documents into structured Document objects with page-level tracking."""
    
    def __init__(self, config: Optional[PDFProcessorConfig] = None) -> None:
        """Initialize PDF processor.
        
        Args:
            config: Configuration settings for PDF processing
        """
        self.config = config or PDFProcessorConfig()
        
        if not PYMUPDF_AVAILABLE:
            logger.error("PyMuPDF is required for PDF processing. Install with: pip install pymupdf")
            raise ImportError("PyMuPDF (fitz) is not installed")
    
    def process(self, pdf_path: Union[str, Path]) -> Document:
        """Process a PDF file into a Document with page-level structure.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Document object with extracted text and metadata
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If PDF cannot be opened or processed
        """
        path = Path(pdf_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {path}")
        
        if not path.suffix.lower() == '.pdf':
            raise ValueError(f"File is not a PDF: {path}")
        
        logger.info("Processing PDF: %s", path)
        
        try:
            with fitz.open(path) as pdf_doc:
                return self._extract_from_pymupdf(pdf_doc, path)
        except Exception as exc:
            logger.error("Failed to process PDF %s: %s", path, exc, exc_info=True)
            raise ValueError(f"Could not process PDF {path}: {exc}") from exc
    
    def _extract_from_pymupdf(self, pdf_doc: Any, path: Path) -> Document:
        """Extract text and structure from PDF using PyMuPDF.
        
        Args:
            pdf_doc: PyMuPDF document object
            path: Original file path
            
        Returns:
            Document with page-level extraction
        """
        # Extract metadata
        metadata = self._extract_metadata(pdf_doc, path) if self.config.extract_metadata else {}
        metadata.update({
            "source": str(path),
            "filename": path.name,
            "file_type": "pdf",
            "file_size": path.stat().st_size,
            "num_pages": len(pdf_doc),
        })
        
        # Extract title
        title = metadata.get("title") or path.stem.replace("_", " ").title()
        
        # Extract pages
        pages: List[DocumentPage] = []
        full_text_parts: List[str] = []
        
        max_pages = self.config.max_pages or len(pdf_doc)
        for page_num in range(min(max_pages, len(pdf_doc))):
            page_obj = pdf_doc[page_num]
            page_text = self._extract_page_text(page_obj, page_num, path)
            
            if page_text.strip():
                pages.append(DocumentPage(
                    page_number=page_num + 1,
                    text=page_text,
                    metadata={
                        "width": page_obj.rect.width,
                        "height": page_obj.rect.height,
                        "rotation": page_obj.rotation,
                    }
                ))
                full_text_parts.append(page_text)
        
        # Generate document ID
        full_text = "\n\n".join(full_text_parts)
        doc_id = self._generate_doc_id(path, full_text)
        
        logger.info("Extracted %d pages from PDF: %s", len(pages), path.name)
        
        return Document(
            doc_id=doc_id,
            title=title,
            text=full_text,
            metadata=metadata,
            pages=pages,
        )
    
    def _extract_page_text(self, page_obj: Any, page_num: int, path: Path) -> str:
        """Extract text from a single PDF page with OCR fallback.
        
        Args:
            page_obj: PyMuPDF page object
            page_num: Page number (0-indexed)
            path: PDF file path (for OCR)
            
        Returns:
            Extracted text content
        """
        # Try direct text extraction
        if self.config.preserve_layout:
            text = page_obj.get_text("text", sort=True)
        else:
            text = page_obj.get_text()
        
        # Check if OCR fallback is needed
        if (self.config.ocr_fallback and 
            OCR_AVAILABLE and 
            len(text.strip()) < self.config.ocr_threshold):
            
            logger.debug("Page %d has minimal text (%d chars), attempting OCR", 
                        page_num + 1, len(text.strip()))
            
            try:
                ocr_text = self._ocr_page(path, page_num)
                if ocr_text and len(ocr_text) > len(text):
                    logger.info("OCR extracted more text for page %d: %d chars vs %d chars",
                               page_num + 1, len(ocr_text), len(text))
                    text = ocr_text
            except Exception as exc:
                logger.warning("OCR failed for page %d: %s", page_num + 1, exc)
        
        return text.strip()
    
    def _ocr_page(self, pdf_path: Path, page_num: int) -> str:
        """Perform OCR on a specific PDF page.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            
        Returns:
            OCR-extracted text
        """
        if not OCR_AVAILABLE:
            return ""
        
        try:
            # Convert single page to image
            images = convert_from_path(
                str(pdf_path),
                first_page=page_num + 1,
                last_page=page_num + 1,
                dpi=self.config.dpi
            )
            
            if images:
                # Perform OCR
                text = pytesseract.image_to_string(images[0])
                return text.strip()
        except Exception as exc:
            logger.warning("OCR conversion failed: %s", exc)
        
        return ""
    
    def _extract_metadata(self, pdf_doc: Any, path: Path) -> Dict[str, Any]:
        """Extract PDF metadata.
        
        Args:
            pdf_doc: PyMuPDF document object
            path: PDF file path
            
        Returns:
            Metadata dictionary
        """
        meta = pdf_doc.metadata or {}
        
        # Clean and normalize metadata
        cleaned = {}
        for key in ["title", "author", "subject", "keywords", "creator", "producer"]:
            value = meta.get(key, "").strip()
            if value:
                cleaned[key] = value
        
        # Add creation/modification dates
        if meta.get("creationDate"):
            cleaned["creation_date"] = meta["creationDate"]
        if meta.get("modDate"):
            cleaned["modification_date"] = meta["modDate"]
        
        return cleaned
    
    @staticmethod
    def _generate_doc_id(path: Path, content: str) -> str:
        """Generate deterministic document ID.
        
        Args:
            path: PDF file path
            content: Extracted text content
            
        Returns:
            Unique document ID
        """
        hasher = hashlib.sha256()
        hasher.update(str(path.name).encode("utf-8"))
        hasher.update(content[:1000].encode("utf-8"))
        digest = hasher.hexdigest()[:12]
        clean_stem = "".join(c if c.isalnum() else "_" for c in path.stem)[:20]
        return f"pdf_{clean_stem}_{digest}"


class PDFBatchProcessor:
    """Process multiple PDF files in batch."""
    
    def __init__(self, config: Optional[PDFProcessorConfig] = None) -> None:
        """Initialize batch processor.
        
        Args:
            config: Configuration for PDF processing
        """
        self.processor = PDFProcessor(config)
    
    def process_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True
    ) -> List[Document]:
        """Process all PDF files in a directory.
        
        Args:
            directory: Directory path
            recursive: Whether to search subdirectories
            
        Returns:
            List of processed Documents
        """
        dir_path = Path(directory).resolve()
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")
        
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(dir_path.glob(pattern))
        
        logger.info("Found %d PDF files in %s", len(pdf_files), dir_path)
        
        documents: List[Document] = []
        for pdf_file in sorted(pdf_files):
            try:
                doc = self.processor.process(pdf_file)
                documents.append(doc)
            except Exception as exc:
                logger.error("Failed to process %s: %s", pdf_file.name, exc)
        
        logger.info("Successfully processed %d/%d PDFs", len(documents), len(pdf_files))
        return documents
    
    def process_files(self, file_paths: List[Union[str, Path]]) -> List[Document]:
        """Process a list of PDF files.
        
        Args:
            file_paths: List of PDF file paths
            
        Returns:
            List of processed Documents
        """
        documents: List[Document] = []
        for file_path in file_paths:
            try:
                doc = self.processor.process(file_path)
                documents.append(doc)
            except Exception as exc:
                logger.error("Failed to process %s: %s", file_path, exc)
        
        return documents
