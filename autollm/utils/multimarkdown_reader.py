from pathlib import Path
from typing import Dict, List, Optional

from llama_index.schema import Document

from autollm.utils.markdown_reader import MarkdownReader


class MultiMarkdownReader(MarkdownReader):
    """
    MultiMarkdown parser.

    Extract text from multiple markdown files. Returns a list of dictionaries with keys as headers and values
    as the text between headers.
    """

    def __init__(self, *args, read_as_single_doc: bool = False, **kwargs) -> None:
        """
        Initialize MultiMarkdownReader.

        Parameters:
            read_as_single_doc (bool): If True, read each markdown as a single document.
        """
        super().__init__(*args, **kwargs)
        self.read_as_single_doc = read_as_single_doc

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        content: Optional[str] = None,
    ) -> List[Document]:
        """Include original_file_path in extra_info and respect read_as_single_doc flag."""
        if extra_info is None:
            extra_info = {}

        if not self.read_as_single_doc:
            # Call parent's load_data method for section-based reading
            return super().load_data(file, extra_info, content)
        # Reading entire markdown as a single document
        with open(file, encoding='utf-8') as f:
            content = f.read()
        if self._remove_hyperlinks:
            content = self.remove_hyperlinks(content)
        if self._remove_images:
            content = self.remove_images(content)
        relative_file_path = str(file)

        return [Document(id_=relative_file_path, text=content, metadata=extra_info)]
