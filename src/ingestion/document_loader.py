"""Document ingestion module for MedFusion-RAFT.

Supports loading documents from plain text, Markdown, JSON, JSONL, and CSV files,
converting them into standardized Document schema objects.
"""

import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, Field

from src.models.schemas import Document

logger = logging.getLogger(__name__)


class DocumentLoaderConfig(BaseModel):
    """Configuration for DocumentLoader."""

    encoding: str = Field(default="utf-8", description="File encoding to use when reading files")
    csv_text_columns: List[str] = Field(
        default=["text", "content", "body", "abstract", "passage", "document"],
        description="Candidate column names for text content in CSV files",
    )
    csv_title_columns: List[str] = Field(
        default=["title", "headline", "name", "subject"],
        description="Candidate column names for title in CSV files",
    )
    csv_id_columns: List[str] = Field(
        default=["id", "doc_id", "pmid", "uid", "identifier"],
        description="Candidate column names for document ID in CSV files",
    )
    ignore_empty_docs: bool = Field(
        default=True,
        description="Whether to skip documents with empty text content",
    )


class DocumentLoader:
    """Dispatches and loads documents from varied file formats into Document objects.

    Supported formats:
        - Plain text (.txt)
        - Markdown (.md, .markdown)
        - JSON (.json)
        - JSON Lines (.jsonl)
        - Delimited text (.csv, .tsv)
    """

    def __init__(self, config: Optional[DocumentLoaderConfig] = None) -> None:
        """Initialize DocumentLoader with configuration.

        Args:
            config: Optional configuration settings. Uses defaults if None.
        """
        self.config = config or DocumentLoaderConfig()
        logger.debug("Initialized DocumentLoader with config: %s", self.config)

    @staticmethod
    def generate_doc_id(file_path: Path, content: str = "", extra: str = "") -> str:
        """Generate a deterministic doc_id from filename and content hash.

        Args:
            file_path: Path to the source file.
            content: Document text content.
            extra: Optional discriminator (e.g. row index).

        Returns:
            Hex string doc_id prefixed with the stem.
        """
        hasher = hashlib.sha256()
        hasher.update(str(file_path.name).encode("utf-8"))
        if extra:
            hasher.update(str(extra).encode("utf-8"))
        if content:
            hasher.update(content[:500].encode("utf-8"))
        digest = hasher.hexdigest()[:12]
        clean_stem = "".join(c if c.isalnum() else "_" for c in file_path.stem)[:20]
        return f"{clean_stem}_{digest}"

    def load(self, file_path: Union[str, Path]) -> List[Document]:
        """Load documents from a single file by dispatching based on extension.

        Args:
            file_path: Path to the target document file.

        Returns:
            List of parsed Document instances.

        Raises:
            FileNotFoundError: If the given file does not exist.
            ValueError: If the file extension is unsupported.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        logger.info("Loading document from: %s (type: %s)", path, suffix)

        if suffix in [".txt", ".text"]:
            return self._load_txt(path)
        elif suffix in [".md", ".markdown"]:
            return self._load_markdown(path)
        elif suffix == ".json":
            return self._load_json(path)
        elif suffix == ".jsonl":
            return self._load_jsonl(path)
        elif suffix in [".csv", ".tsv"]:
            return self._load_csv(path, delimiter="\t" if suffix == ".tsv" else ",")
        else:
            raise ValueError(f"Unsupported file format: {suffix} for path {path}")

    def load_directory(
        self,
        dir_path: Union[str, Path],
        extensions: Optional[Sequence[str]] = None,
        recursive: bool = True,
    ) -> List[Document]:
        """Load all matching documents from a directory.

        Args:
            dir_path: Directory path to scan.
            extensions: Allowed extensions (e.g. ['.txt', '.json']). If None, all supported are loaded.
            recursive: Whether to search subdirectories recursively.

        Returns:
            Aggregated list of Document objects from all valid files.
        """
        path = Path(dir_path).resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"Directory not found: {path}")

        valid_exts = {ext.lower() for ext in extensions} if extensions else {
            ".txt", ".text", ".md", ".markdown", ".json", ".jsonl", ".csv", ".tsv"
        }

        pattern = "**/*" if recursive else "*"
        documents: List[Document] = []

        for candidate in sorted(path.glob(pattern)):
            if candidate.is_file() and candidate.suffix.lower() in valid_exts:
                try:
                    docs = self.load(candidate)
                    documents.extend(docs)
                except Exception as exc:
                    logger.warning("Failed to load file %s: %s", candidate, exc, exc_info=True)

        logger.info("Loaded %d total documents from directory %s", len(documents), dir_path)
        return documents

    def _read_file_text(self, path: Path) -> str:
        """Safely read text with fallback encoding handling."""
        try:
            return path.read_text(encoding=self.config.encoding)
        except UnicodeDecodeError:
            logger.warning("Encoding error reading %s with %s; falling back to replace", path, self.config.encoding)
            return path.read_text(encoding=self.config.encoding, errors="replace")

    def _load_txt(self, path: Path) -> List[Document]:
        """Load a plain text file into a Document."""
        raw_text = self._read_file_text(path).strip()
        if self.config.ignore_empty_docs and not raw_text:
            logger.debug("Skipping empty txt file: %s", path)
            return []

        doc_id = self.generate_doc_id(path, raw_text)
        title = path.stem.replace("_", " ").title()

        return [
            Document(
                doc_id=doc_id,
                title=title,
                text=raw_text,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "file_type": "txt",
                    "file_size": path.stat().st_size,
                },
                pages=None,
            )
        ]

    def _load_markdown(self, path: Path) -> List[Document]:
        """Load a markdown file, extracting top header if available."""
        raw_text = self._read_file_text(path).strip()
        if self.config.ignore_empty_docs and not raw_text:
            logger.debug("Skipping empty markdown file: %s", path)
            return []

        title = path.stem.replace("_", " ").title()
        for line in raw_text.splitlines():
            line_s = line.strip()
            if line_s.startswith("# "):
                title = line_s.lstrip("# ").strip()
                break

        doc_id = self.generate_doc_id(path, raw_text)
        return [
            Document(
                doc_id=doc_id,
                title=title,
                text=raw_text,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "file_type": "markdown",
                    "file_size": path.stat().st_size,
                },
                pages=None,
            )
        ]

    def _load_json(self, path: Path) -> List[Document]:
        """Load JSON file containing a single document object or list of documents."""
        raw_content = self._read_file_text(path)
        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError as err:
            logger.error("JSON decode error in %s: %s", path, err)
            raise

        records = data if isinstance(data, list) else [data]
        documents: List[Document] = []

        for idx, item in enumerate(records):
            if not isinstance(item, dict):
                continue
            doc = self._dict_to_document(item, path, idx)
            if doc:
                documents.append(doc)

        return documents

    def _load_jsonl(self, path: Path) -> List[Document]:
        """Load a JSON Lines (.jsonl) file."""
        documents: List[Document] = []
        raw_content = self._read_file_text(path)

        for idx, line in enumerate(raw_content.splitlines()):
            line_s = line.strip()
            if not line_s:
                continue
            try:
                item = json.loads(line_s)
                if isinstance(item, dict):
                    doc = self._dict_to_document(item, path, idx)
                    if doc:
                        documents.append(doc)
            except json.JSONDecodeError as err:
                logger.warning("Skipping invalid JSON line %d in %s: %s", idx, path, err)

        return documents

    def _load_csv(self, path: Path, delimiter: str = ",") -> List[Document]:
        """Load CSV or TSV file extracting text, title, and metadata."""
        documents: List[Document] = []
        raw_text = self._read_file_text(path)
        lines = [ln for ln in raw_text.splitlines() if ln.strip()]
        if not lines:
            return []

        reader = csv.DictReader(lines, delimiter=delimiter)
        for idx, row in enumerate(reader):
            doc = self._dict_to_document(row, path, idx)
            if doc:
                documents.append(doc)

        return documents

    def _dict_to_document(
        self, data: Dict[str, Any], path: Path, index: int
    ) -> Optional[Document]:
        """Convert a dictionary record from JSON/CSV into a Document instance."""
        # Find text field
        text: Optional[str] = None
        for key in self.config.csv_text_columns:
            if key in data and data[key]:
                text = str(data[key]).strip()
                break
        if text is None and "text" in data:
            text = str(data["text"]).strip()

        if self.config.ignore_empty_docs and not text:
            return None

        # Find title field
        title = ""
        for key in self.config.csv_title_columns:
            if key in data and data[key]:
                title = str(data[key]).strip()
                break
        if not title:
            title = f"{path.stem} - Record {index + 1}"

        # Find or generate doc_id
        doc_id: Optional[str] = None
        for key in self.config.csv_id_columns:
            if key in data and data[key]:
                doc_id = str(data[key]).strip()
                break
        if not doc_id:
            doc_id = self.generate_doc_id(path, text or "", extra=str(index))

        # Separate custom metadata
        metadata: Dict[str, Any] = {
            "source": str(path),
            "filename": path.name,
            "record_index": index,
        }
        for k, v in data.items():
            if k not in (self.config.csv_text_columns + self.config.csv_title_columns + self.config.csv_id_columns):
                metadata[k] = v

        pages = data.get("pages")
        if pages is not None and not isinstance(pages, list):
            pages = [str(pages)]

        return Document(
            doc_id=doc_id,
            title=title,
            text=text or "",
            metadata=metadata,
            pages=pages,
        )
