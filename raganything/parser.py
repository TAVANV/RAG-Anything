# type: ignore
"""
Generic Document Parser Utility

This module provides functionality for parsing PDF and image documents using MinerU 2.0 library,
and converts the parsing results into markdown and JSON formats

Note: MinerU 2.0 no longer includes LibreOffice document conversion module.
For Office documents (.doc, .docx, .ppt, .pptx), please convert them to PDF format first.
"""

from __future__ import annotations


import json
import argparse
import base64
import subprocess
import tempfile
import logging
import threading
import time
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Union,
    Tuple,
    Any,
    TypeVar,
)

T = TypeVar("T")

# 全局 LibreOffice 转换信号量（限制并发数，防止冲突）
# LibreOffice headless 模式对并发有限制，使用信号量控制最大并发数
# 经验值：macOS/Linux 建议 1-2，Windows 建议 1
import os
_max_concurrent_libreoffice = int(os.getenv("LIBREOFFICE_MAX_CONCURRENT", "1"))
_libreoffice_semaphore = threading.Semaphore(_max_concurrent_libreoffice)


class MineruExecutionError(Exception):
    """catch mineru error"""

    def __init__(self, return_code, error_msg):
        self.return_code = return_code
        self.error_msg = error_msg
        super().__init__(
            f"Mineru command failed with return code {return_code}: {error_msg}"
        )


class Parser:
    """
    Base class for document parsing utilities.

    Defines common functionality and constants for parsing different document types.
    """

    # Define common file formats
    OFFICE_FORMATS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
    IMAGE_FORMATS = {".png", ".jpeg", ".jpg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
    TEXT_FORMATS = {".txt", ".md"}

    # Class-level logger
    logger = logging.getLogger(__name__)

    def __init__(self) -> None:
        """Initialize the base parser."""
        pass

    @staticmethod
    def convert_office_to_pdf(
        doc_path: Union[str, Path], output_dir: Optional[str] = None
    ) -> Path:
        """
        Convert Office document (.doc, .docx, .ppt, .pptx, .xls, .xlsx) to PDF.
        Requires LibreOffice to be installed.

        Args:
            doc_path: Path to the Office document file
            output_dir: Output directory for the PDF file

        Returns:
            Path to the generated PDF file
        """
        try:
            # Convert to Path object for easier handling
            doc_path = Path(doc_path)
            if not doc_path.exists():
                raise FileNotFoundError(f"Office document does not exist: {doc_path}")

            name_without_suff = doc_path.stem

            # Prepare output directory
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = doc_path.parent / "libreoffice_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)

            # Create temporary directory for PDF conversion
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Convert to PDF using LibreOffice (使用信号量限制并发数)
                logging.info(f"Converting {doc_path.name} to PDF using LibreOffice...")

                # 使用信号量限制 LibreOffice 并发数（防止配置文件锁冲突）
                with _libreoffice_semaphore:
                    logging.debug(f"Acquired LibreOffice semaphore for {doc_path.name} "
                                f"(max concurrent: {_max_concurrent_libreoffice})")

                    # Prepare subprocess parameters to hide console window on Windows
                    import platform

                    # Try LibreOffice commands in order of preference
                    commands_to_try = ["libreoffice", "soffice"]

                    conversion_successful = False
                    for cmd in commands_to_try:
                        try:
                            # Create a unique temporary user profile for this conversion
                            # This prevents lock file conflicts and state pollution between conversions
                            with tempfile.TemporaryDirectory() as user_profile_dir:
                                convert_cmd = [
                                    cmd,
                                    "--headless",
                                    "--convert-to",
                                    "pdf",
                                    "--outdir",
                                    str(temp_path),
                                    "-env:UserInstallation=file://" + str(Path(user_profile_dir).absolute()),
                                    str(doc_path),
                                ]

                                # Prepare conversion subprocess parameters
                                convert_subprocess_kwargs = {
                                    "capture_output": True,
                                    "text": True,
                                    "timeout": 60,  # 60 second timeout
                                    "encoding": "utf-8",
                                    "errors": "ignore",
                                }

                                # Hide console window on Windows
                                if platform.system() == "Windows":
                                    convert_subprocess_kwargs["creationflags"] = (
                                        subprocess.CREATE_NO_WINDOW
                                    )

                                result = subprocess.run(
                                    convert_cmd, **convert_subprocess_kwargs
                                )

                                if result.returncode == 0:
                                    conversion_successful = True
                                    logging.info(
                                        f"Successfully converted {doc_path.name} to PDF using {cmd}\n"
                                        f"  stdout: {result.stdout}\n"
                                        f"  stderr: {result.stderr}"
                                    )
                                    break
                                else:
                                    logging.warning(
                                        f"LibreOffice command '{cmd}' failed (returncode={result.returncode}):\n"
                                        f"  stdout: {result.stdout}\n"
                                        f"  stderr: {result.stderr}"
                                    )
                        except FileNotFoundError:
                            logging.warning(f"LibreOffice command '{cmd}' not found")
                        except subprocess.TimeoutExpired:
                            logging.warning(f"LibreOffice command '{cmd}' timed out")
                        except Exception as e:
                            logging.error(
                                f"LibreOffice command '{cmd}' failed with exception: {e}"
                            )

                    if not conversion_successful:
                        raise RuntimeError(
                            f"LibreOffice conversion failed for {doc_path.name}. "
                            f"Please ensure LibreOffice is installed:\n"
                            "- Windows: Download from https://www.libreoffice.org/download/download/\n"
                            "- macOS: brew install --cask libreoffice\n"
                            "- Ubuntu/Debian: sudo apt-get install libreoffice\n"
                            "- CentOS/RHEL: sudo yum install libreoffice\n"
                            "Alternatively, convert the document to PDF manually."
                        )

                # Find the generated PDF
                pdf_files = list(temp_path.glob("*.pdf"))
                if not pdf_files:
                    # 添加详细的调试信息
                    all_files = list(temp_path.glob("*"))
                    logging.error(
                        f"PDF conversion failed for {doc_path.name}:\n"
                        f"  - Expected PDF in: {temp_path}\n"
                        f"  - Files found: {[f.name for f in all_files]}\n"
                        f"  - Source file exists: {doc_path.exists()}\n"
                        f"  - Source file size: {doc_path.stat().st_size if doc_path.exists() else 'N/A'}"
                    )
                    raise RuntimeError(
                        f"PDF conversion failed for {doc_path.name} - no PDF file generated. "
                        f"Please check LibreOffice installation or try manual conversion."
                    )

                pdf_path = pdf_files[0]
                logging.info(
                    f"Generated PDF: {pdf_path.name} ({pdf_path.stat().st_size} bytes)"
                )

                # Validate the generated PDF
                if pdf_path.stat().st_size < 100:  # Very small file, likely empty
                    raise RuntimeError(
                        "Generated PDF appears to be empty or corrupted. "
                        "Original file may have issues or LibreOffice conversion failed."
                    )

                # Copy PDF to final output directory
                final_pdf_path = base_output_dir / f"{name_without_suff}.pdf"
                import shutil

                shutil.copy2(pdf_path, final_pdf_path)

                return final_pdf_path

        except Exception as e:
            logging.error(f"Error in convert_office_to_pdf: {str(e)}")
            raise

    @staticmethod
    def convert_text_to_pdf(
        text_path: Union[str, Path], output_dir: Optional[str] = None
    ) -> Path:
        """
        Convert text file (.txt, .md) to PDF using ReportLab with full markdown support.

        Args:
            text_path: Path to the text file
            output_dir: Output directory for the PDF file

        Returns:
            Path to the generated PDF file
        """
        try:
            text_path = Path(text_path)
            if not text_path.exists():
                raise FileNotFoundError(f"Text file does not exist: {text_path}")

            # Supported text formats
            supported_text_formats = {".txt", ".md"}
            if text_path.suffix.lower() not in supported_text_formats:
                raise ValueError(f"Unsupported text format: {text_path.suffix}")

            # Read the text content
            try:
                with open(text_path, "r", encoding="utf-8") as f:
                    text_content = f.read()
            except UnicodeDecodeError:
                # Try with different encodings
                for encoding in ["gbk", "latin-1", "cp1252"]:
                    try:
                        with open(text_path, "r", encoding=encoding) as f:
                            text_content = f.read()
                        logging.info(f"Successfully read file with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise RuntimeError(
                        f"Could not decode text file {text_path.name} with any supported encoding"
                    )

            # Prepare output directory
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = text_path.parent / "reportlab_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = base_output_dir / f"{text_path.stem}.pdf"

            # Convert text to PDF
            logging.info(f"Converting {text_path.name} to PDF...")

            try:
                from reportlab.lib.pagesizes import A4
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                from reportlab.pdfbase import pdfmetrics
                from reportlab.pdfbase.ttfonts import TTFont

                support_chinese = True
                try:
                    if "WenQuanYi" not in pdfmetrics.getRegisteredFontNames():
                        if not Path(
                            "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc"
                        ).exists():
                            support_chinese = False
                            logging.warning(
                                "WenQuanYi font not found at /usr/share/fonts/wqy-microhei/wqy-microhei.ttc. Chinese characters may not render correctly."
                            )
                        else:
                            pdfmetrics.registerFont(
                                TTFont(
                                    "WenQuanYi",
                                    "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
                                )
                            )
                except Exception as e:
                    support_chinese = False
                    logging.warning(
                        f"Failed to register WenQuanYi font: {e}. Chinese characters may not render correctly."
                    )

                # Create PDF document
                doc = SimpleDocTemplate(
                    str(pdf_path),
                    pagesize=A4,
                    leftMargin=inch,
                    rightMargin=inch,
                    topMargin=inch,
                    bottomMargin=inch,
                )

                # Get styles
                styles = getSampleStyleSheet()
                normal_style = styles["Normal"]
                heading_style = styles["Heading1"]
                if support_chinese:
                    normal_style.fontName = "WenQuanYi"
                    heading_style.fontName = "WenQuanYi"

                # Try to register a font that supports Chinese characters
                try:
                    # Try to use system fonts that support Chinese
                    import platform

                    system = platform.system()
                    if system == "Windows":
                        # Try common Windows fonts
                        for font_name in ["SimSun", "SimHei", "Microsoft YaHei"]:
                            try:
                                from reportlab.pdfbase.cidfonts import (
                                    UnicodeCIDFont,
                                )

                                pdfmetrics.registerFont(UnicodeCIDFont(font_name))
                                normal_style.fontName = font_name
                                heading_style.fontName = font_name
                                break
                            except Exception:
                                continue
                    elif system == "Darwin":  # macOS
                        for font_name in ["STSong-Light", "STHeiti"]:
                            try:
                                from reportlab.pdfbase.cidfonts import (
                                    UnicodeCIDFont,
                                )

                                pdfmetrics.registerFont(UnicodeCIDFont(font_name))
                                normal_style.fontName = font_name
                                heading_style.fontName = font_name
                                break
                            except Exception:
                                continue
                except Exception:
                    pass  # Use default fonts if Chinese font setup fails

                # Build content
                story = []

                # Handle markdown or plain text
                if text_path.suffix.lower() == ".md":
                    # Handle markdown content - simplified implementation
                    lines = text_content.split("\n")
                    for line in lines:
                        line = line.strip()
                        if not line:
                            story.append(Spacer(1, 12))
                            continue

                        # Headers
                        if line.startswith("#"):
                            level = len(line) - len(line.lstrip("#"))
                            header_text = line.lstrip("#").strip()
                            if header_text:
                                header_style = ParagraphStyle(
                                    name=f"Heading{level}",
                                    parent=heading_style,
                                    fontSize=max(16 - level, 10),
                                    spaceAfter=8,
                                    spaceBefore=16 if level <= 2 else 12,
                                )
                                story.append(Paragraph(header_text, header_style))
                        else:
                            # Regular text
                            story.append(Paragraph(line, normal_style))
                            story.append(Spacer(1, 6))
                else:
                    # Handle plain text files (.txt)
                    logging.info(
                        f"Processing plain text file with {len(text_content)} characters..."
                    )

                    # Split text into lines and process each line
                    lines = text_content.split("\n")
                    line_count = 0

                    for line in lines:
                        line = line.rstrip()
                        line_count += 1

                        # Empty lines
                        if not line.strip():
                            story.append(Spacer(1, 6))
                            continue

                        # Regular text lines
                        # Escape special characters for ReportLab
                        safe_line = (
                            line.replace("&", "&amp;")
                            .replace("<", "&lt;")
                            .replace(">", "&gt;")
                        )

                        # Create paragraph
                        story.append(Paragraph(safe_line, normal_style))
                        story.append(Spacer(1, 3))

                    logging.info(f"Added {line_count} lines to PDF")

                    # If no content was added, add a placeholder
                    if not story:
                        story.append(Paragraph("(Empty text file)", normal_style))

                # Build PDF
                doc.build(story)
                logging.info(
                    f"Successfully converted {text_path.name} to PDF ({pdf_path.stat().st_size / 1024:.1f} KB)"
                )

            except ImportError:
                raise RuntimeError(
                    "reportlab is required for text-to-PDF conversion. "
                    "Please install it using: pip install reportlab"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to convert text file {text_path.name} to PDF: {str(e)}"
                )

            # Validate the generated PDF
            if not pdf_path.exists() or pdf_path.stat().st_size < 100:
                raise RuntimeError(
                    f"PDF conversion failed for {text_path.name} - generated PDF is empty or corrupted."
                )

            return pdf_path

        except Exception as e:
            logging.error(f"Error in convert_text_to_pdf: {str(e)}")
            raise

    @staticmethod
    def _process_inline_markdown(text: str) -> str:
        """
        Process inline markdown formatting (bold, italic, code, links)

        Args:
            text: Raw text with markdown formatting

        Returns:
            Text with ReportLab markup
        """
        import re

        # Escape special characters for ReportLab
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # Bold text: **text** or __text__
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"__(.*?)__", r"<b>\1</b>", text)

        # Italic text: *text* or _text_ (but not in the middle of words)
        text = re.sub(r"(?<!\w)\*([^*\n]+?)\*(?!\w)", r"<i>\1</i>", text)
        text = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"<i>\1</i>", text)

        # Inline code: `code`
        text = re.sub(
            r"`([^`]+?)`",
            r'<font name="Courier" size="9" color="darkred">\1</font>',
            text,
        )

        # Links: [text](url) - convert to text with URL annotation
        def link_replacer(match):
            link_text = match.group(1)
            url = match.group(2)
            return f'<link href="{url}" color="blue"><u>{link_text}</u></link>'

        text = re.sub(r"\[([^\]]+?)\]\(([^)]+?)\)", link_replacer, text)

        # Strikethrough: ~~text~~
        text = re.sub(r"~~(.*?)~~", r"<strike>\1</strike>", text)

        return text

    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[str] = None,
        method: str = "auto",
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Abstract method to parse PDF document.
        Must be implemented by subclasses.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Output directory path
            method: Parsing method (auto, txt, ocr)
            lang: Document language for OCR optimization
            **kwargs: Additional parameters for parser-specific command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        raise NotImplementedError("parse_pdf must be implemented by subclasses")

    def parse_image(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Abstract method to parse image document.
        Must be implemented by subclasses.

        Note: Different parsers may support different image formats.
        Check the specific parser's documentation for supported formats.

        Args:
            image_path: Path to the image file
            output_dir: Output directory path
            lang: Document language for OCR optimization
            **kwargs: Additional parameters for parser-specific command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        raise NotImplementedError("parse_image must be implemented by subclasses")

    def parse_document(
        self,
        file_path: Union[str, Path],
        method: str = "auto",
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Abstract method to parse a document.
        Must be implemented by subclasses.

        Args:
            file_path: Path to the file to be parsed
            method: Parsing method (auto, txt, ocr)
            output_dir: Output directory path
            lang: Document language for OCR optimization
            **kwargs: Additional parameters for parser-specific command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        raise NotImplementedError("parse_document must be implemented by subclasses")

    def check_installation(self) -> bool:
        """
        Abstract method to check if the parser is properly installed.
        Must be implemented by subclasses.

        Returns:
            bool: True if installation is valid, False otherwise
        """
        raise NotImplementedError(
            "check_installation must be implemented by subclasses"
        )


class MineruParser(Parser):
    """
    MinerU 2.0 document parsing utility class

    Supports parsing PDF and image documents, converting the content into structured data
    and generating markdown and JSON output.

    Note: Office documents are no longer directly supported. Please convert them to PDF first.
    """

    __slots__ = ()

    # Class-level logger
    logger = logging.getLogger(__name__)

    def __init__(self) -> None:
        """Initialize MineruParser"""
        super().__init__()

    @staticmethod
    def _run_mineru_command(
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        method: str = "auto",
        lang: Optional[str] = None,
        backend: Optional[str] = None,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        formula: bool = True,
        table: bool = True,
        device: Optional[str] = None,
        source: Optional[str] = None,
        vlm_url: Optional[str] = None,
    ) -> None:
        """
        Run mineru command line tool

        Args:
            input_path: Path to input file or directory
            output_dir: Output directory path
            method: Parsing method (auto, txt, ocr)
            lang: Document language for OCR optimization
            backend: Parsing backend
            start_page: Starting page number (0-based)
            end_page: Ending page number (0-based)
            formula: Enable formula parsing
            table: Enable table parsing
            device: Inference device
            source: Model source
            vlm_url: When the backend is `vlm-sglang-client`, you need to specify the server_url
        """
        cmd = [
            "mineru",
            "-p",
            str(input_path),
            "-o",
            str(output_dir),
            "-m",
            method,
        ]

        if backend:
            cmd.extend(["-b", backend])
        if source:
            cmd.extend(["--source", source])
        if lang:
            cmd.extend(["-l", lang])
        if start_page is not None:
            cmd.extend(["-s", str(start_page)])
        if end_page is not None:
            cmd.extend(["-e", str(end_page)])
        if not formula:
            cmd.extend(["-f", "false"])
        if not table:
            cmd.extend(["-t", "false"])
        if device:
            cmd.extend(["-d", device])
        if vlm_url:
            cmd.extend(["-u", vlm_url])

        output_lines = []
        error_lines = []

        try:
            # Prepare subprocess parameters to hide console window on Windows
            import platform
            import threading
            from queue import Queue, Empty

            # Log the command being executed
            logging.info(f"Executing mineru command: {' '.join(cmd)}")

            subprocess_kwargs = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "encoding": "utf-8",
                "errors": "ignore",
                "bufsize": 1,  # Line buffered
            }

            # Hide console window on Windows
            if platform.system() == "Windows":
                subprocess_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            # Function to read output from subprocess and add to queue
            def enqueue_output(pipe, queue, prefix):
                try:
                    for line in iter(pipe.readline, ""):
                        if line.strip():  # Only add non-empty lines
                            queue.put((prefix, line.strip()))
                    pipe.close()
                except Exception as e:
                    queue.put((prefix, f"Error reading {prefix}: {e}"))

            # Start subprocess
            process = subprocess.Popen(cmd, **subprocess_kwargs)

            # Create queues for stdout and stderr
            stdout_queue = Queue()
            stderr_queue = Queue()

            # Start threads to read output
            stdout_thread = threading.Thread(
                target=enqueue_output, args=(process.stdout, stdout_queue, "STDOUT")
            )
            stderr_thread = threading.Thread(
                target=enqueue_output, args=(process.stderr, stderr_queue, "STDERR")
            )

            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()

            # Process output in real time
            while process.poll() is None:
                # Check stdout queue
                try:
                    while True:
                        prefix, line = stdout_queue.get_nowait()
                        output_lines.append(line)
                        # Log mineru output with INFO level, prefixed with [MinerU]
                        logging.info(f"[MinerU] {line}")
                except Empty:
                    pass

                # Check stderr queue
                try:
                    while True:
                        prefix, line = stderr_queue.get_nowait()
                        # Log mineru errors with WARNING level
                        if "warning" in line.lower():
                            logging.warning(f"[MinerU] {line}")
                        elif "error" in line.lower():
                            logging.error(f"[MinerU] {line}")
                            error_message = line.split("\n")[0]
                            error_lines.append(error_message)
                        else:
                            logging.info(f"[MinerU] {line}")
                except Empty:
                    pass

                # Small delay to prevent busy waiting
                import time

                time.sleep(0.1)

            # Process any remaining output after process completion
            try:
                while True:
                    prefix, line = stdout_queue.get_nowait()
                    output_lines.append(line)
                    logging.info(f"[MinerU] {line}")
            except Empty:
                pass

            try:
                while True:
                    prefix, line = stderr_queue.get_nowait()
                    if "warning" in line.lower():
                        logging.warning(f"[MinerU] {line}")
                    elif "error" in line.lower():
                        logging.error(f"[MinerU] {line}")
                        error_message = line.split("\n")[0]
                        error_lines.append(error_message)
                    else:
                        logging.info(f"[MinerU] {line}")
            except Empty:
                pass

            # Wait for process to complete and get return code
            return_code = process.wait()

            # Wait for threads to finish
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)

            if return_code != 0 or error_lines:
                logging.info("[MinerU] Command executed failed")
                raise MineruExecutionError(return_code, error_lines)
            else:
                logging.info("[MinerU] Command executed successfully")

        except MineruExecutionError:
            raise
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running mineru subprocess command: {e}")
            logging.error(f"Command: {' '.join(cmd)}")
            logging.error(f"Return code: {e.returncode}")
            raise
        except FileNotFoundError:
            raise RuntimeError(
                "mineru command not found. Please ensure MinerU 2.0 is properly installed:\n"
                "pip install -U 'mineru[core]' or uv pip install -U 'mineru[core]'"
            )
        except Exception as e:
            error_message = f"Unexpected error running mineru command: {e}"
            logging.error(error_message)
            raise RuntimeError(error_message) from e

    @staticmethod
    def _read_output_files(
        output_dir: Path, file_stem: str, method: str = "auto"
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Read the output files generated by mineru

        Args:
            output_dir: Output directory
            file_stem: File name without extension

        Returns:
            Tuple containing (content list JSON, Markdown text)
        """
        # Look for the generated files
        md_file = output_dir / f"{file_stem}.md"
        json_file = output_dir / f"{file_stem}_content_list.json"
        images_base_dir = output_dir  # Base directory for images

        file_stem_subdir = output_dir / file_stem
        if file_stem_subdir.exists():
            md_file = file_stem_subdir / method / f"{file_stem}.md"
            json_file = file_stem_subdir / method / f"{file_stem}_content_list.json"
            images_base_dir = file_stem_subdir / method

        # Read markdown content
        md_content = ""
        if md_file.exists():
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    md_content = f.read()
            except Exception as e:
                logging.warning(f"Could not read markdown file {md_file}: {e}")

        # Read JSON content list
        content_list = []
        if json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    content_list = json.load(f)

                # Always fix relative paths in content_list to absolute paths
                logging.info(
                    f"Fixing image paths in {json_file} with base directory: {images_base_dir}"
                )
                for item in content_list:
                    if isinstance(item, dict):
                        for field_name in [
                            "img_path",
                            "table_img_path",
                            "equation_img_path",
                        ]:
                            if field_name in item and item[field_name]:
                                img_path = item[field_name]
                                absolute_img_path = (
                                    images_base_dir / img_path
                                ).resolve()
                                item[field_name] = str(absolute_img_path)
                                logging.debug(
                                    f"Updated {field_name}: {img_path} -> {item[field_name]}"
                                )

            except Exception as e:
                logging.warning(f"Could not read JSON file {json_file}: {e}")

        return content_list, md_content

    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[str] = None,
        method: str = "auto",
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Parse PDF document using MinerU 2.0

        Args:
            pdf_path: Path to the PDF file
            output_dir: Output directory path
            method: Parsing method (auto, txt, ocr)
            lang: Document language for OCR optimization
            **kwargs: Additional parameters for mineru command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        try:
            # Convert to Path object for easier handling
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

            name_without_suff = pdf_path.stem

            # Prepare output directory
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = pdf_path.parent / "mineru_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)

            # Run mineru command
            self._run_mineru_command(
                input_path=pdf_path,
                output_dir=base_output_dir,
                method=method,
                lang=lang,
                **kwargs,
            )

            # Read the generated output files
            backend = kwargs.get("backend", "")
            if backend.startswith("vlm-"):
                method = "vlm"

            content_list, _ = self._read_output_files(
                base_output_dir, name_without_suff, method=method
            )
            return content_list

        except MineruExecutionError:
            raise
        except Exception as e:
            logging.error(f"Error in parse_pdf: {str(e)}")
            raise

    def parse_image(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Parse image document using MinerU 2.0

        Note: MinerU 2.0 natively supports .png, .jpeg, .jpg formats.
        Other formats (.bmp, .tiff, .tif, etc.) will be automatically converted to .png.

        Args:
            image_path: Path to the image file
            output_dir: Output directory path
            lang: Document language for OCR optimization
            **kwargs: Additional parameters for mineru command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        try:
            # Convert to Path object for easier handling
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file does not exist: {image_path}")

            # Supported image formats by MinerU 2.0
            mineru_supported_formats = {".png", ".jpeg", ".jpg"}

            # All supported image formats (including those we can convert)
            all_supported_formats = {
                ".png",
                ".jpeg",
                ".jpg",
                ".bmp",
                ".tiff",
                ".tif",
                ".gif",
                ".webp",
            }

            ext = image_path.suffix.lower()
            if ext not in all_supported_formats:
                raise ValueError(
                    f"Unsupported image format: {ext}. Supported formats: {', '.join(all_supported_formats)}"
                )

            # Determine the actual image file to process
            actual_image_path = image_path
            temp_converted_file = None

            # If format is not natively supported by MinerU, convert it
            if ext not in mineru_supported_formats:
                logging.info(
                    f"Converting {ext} image to PNG for MinerU compatibility..."
                )

                try:
                    from PIL import Image
                except ImportError:
                    raise RuntimeError(
                        "PIL/Pillow is required for image format conversion. "
                        "Please install it using: pip install Pillow"
                    )

                # Create temporary directory for conversion
                temp_dir = Path(tempfile.mkdtemp())
                temp_converted_file = temp_dir / f"{image_path.stem}_converted.png"

                try:
                    # Open and convert image
                    with Image.open(image_path) as img:
                        # Handle different image modes
                        if img.mode in ("RGBA", "LA", "P"):
                            # For images with transparency or palette, convert to RGB first
                            if img.mode == "P":
                                img = img.convert("RGBA")

                            # Create white background for transparent images
                            background = Image.new("RGB", img.size, (255, 255, 255))
                            if img.mode == "RGBA":
                                background.paste(
                                    img, mask=img.split()[-1]
                                )  # Use alpha channel as mask
                            else:
                                background.paste(img)
                            img = background
                        elif img.mode not in ("RGB", "L"):
                            # Convert other modes to RGB
                            img = img.convert("RGB")

                        # Save as PNG
                        img.save(temp_converted_file, "PNG", optimize=True)
                        logging.info(
                            f"Successfully converted {image_path.name} to PNG ({temp_converted_file.stat().st_size / 1024:.1f} KB)"
                        )

                        actual_image_path = temp_converted_file

                except Exception as e:
                    if temp_converted_file and temp_converted_file.exists():
                        temp_converted_file.unlink()
                    raise RuntimeError(
                        f"Failed to convert image {image_path.name}: {str(e)}"
                    )

            name_without_suff = image_path.stem

            # Prepare output directory
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = image_path.parent / "mineru_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Run mineru command (images are processed with OCR method)
                self._run_mineru_command(
                    input_path=actual_image_path,
                    output_dir=base_output_dir,
                    method="ocr",  # Images require OCR method
                    lang=lang,
                    **kwargs,
                )

                # Read the generated output files
                content_list, _ = self._read_output_files(
                    base_output_dir, name_without_suff, method="ocr"
                )
                return content_list

            except MineruExecutionError:
                raise

            finally:
                # Clean up temporary converted file if it was created
                if temp_converted_file and temp_converted_file.exists():
                    try:
                        temp_converted_file.unlink()
                        temp_converted_file.parent.rmdir()  # Remove temp directory if empty
                    except Exception:
                        pass  # Ignore cleanup errors

        except Exception as e:
            logging.error(f"Error in parse_image: {str(e)}")
            raise

    def parse_office_doc(
        self,
        doc_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Parse office document by first converting to PDF, then parsing with MinerU 2.0

        Note: This method requires LibreOffice to be installed separately for PDF conversion.
        MinerU 2.0 no longer includes built-in Office document conversion.

        Supported formats: .doc, .docx, .ppt, .pptx, .xls, .xlsx

        Args:
            doc_path: Path to the document file (.doc, .docx, .ppt, .pptx, .xls, .xlsx)
            output_dir: Output directory path
            lang: Document language for OCR optimization
            **kwargs: Additional parameters for mineru command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        try:
            # Convert Office document to PDF using base class method
            pdf_path = self.convert_office_to_pdf(doc_path, output_dir)

            # Parse the converted PDF
            return self.parse_pdf(
                pdf_path=pdf_path, output_dir=output_dir, lang=lang, **kwargs
            )

        except Exception as e:
            logging.error(f"Error in parse_office_doc: {str(e)}")
            raise

    def parse_text_file(
        self,
        text_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Parse text file directly without PDF conversion (optimized for RAG)

        Supported formats: .txt, .md

        Args:
            text_path: Path to the text file (.txt, .md)
            output_dir: Output directory path
            lang: Document language (unused, kept for API compatibility)
            **kwargs: Additional parameters (unused, kept for API compatibility)

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        # Suppress unused parameter warnings - kept for API compatibility
        _ = lang, kwargs

        try:
            text_path = Path(text_path)
            if not text_path.exists():
                raise FileNotFoundError(f"Text file does not exist: {text_path}")

            # Read text content with multiple encoding support
            text_content = None
            for encoding in ["utf-8", "gbk", "latin-1", "cp1252"]:
                try:
                    with open(text_path, "r", encoding=encoding) as f:
                        text_content = f.read()
                    logging.info(f"Successfully read {text_path.name} with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue

            if text_content is None:
                raise RuntimeError(
                    f"Could not decode text file {text_path.name} with any supported encoding"
                )

            # Construct content_list in MinerU-compatible format
            content_list = [{
                "type": "text",
                "text": text_content,
                "page_idx": 0  # Text files are treated as single page
            }]

            # Optionally save to JSON for consistency with MinerU output
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                json_output = output_path / f"{text_path.stem}_content_list.json"
                with open(json_output, "w", encoding="utf-8") as f:
                    json.dump(content_list, f, ensure_ascii=False, indent=2)

                logging.info(f"Saved content_list to: {json_output}")

            logging.info(
                f"Successfully parsed text file: {text_path.name} "
                f"({len(text_content)} characters)"
            )

            return content_list

        except Exception as e:
            logging.error(f"Error in parse_text_file: {str(e)}")
            raise

    def parse_document(
        self,
        file_path: Union[str, Path],
        method: str = "auto",
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Parse document using MinerU 2.0 based on file extension

        Args:
            file_path: Path to the file to be parsed
            method: Parsing method (auto, txt, ocr)
            output_dir: Output directory path
            lang: Document language for OCR optimization
            **kwargs: Additional parameters for mineru command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        # Convert to Path object
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        # Get file extension
        ext = file_path.suffix.lower()

        # Choose appropriate parser based on file type
        if ext == ".pdf":
            return self.parse_pdf(file_path, output_dir, method, lang, **kwargs)
        elif ext in self.IMAGE_FORMATS:
            return self.parse_image(file_path, output_dir, lang, **kwargs)
        elif ext in self.OFFICE_FORMATS:
            logging.warning(
                f"Warning: Office document detected ({ext}). "
                f"MinerU 2.0 requires conversion to PDF first."
            )
            return self.parse_office_doc(file_path, output_dir, lang, **kwargs)
        elif ext in self.TEXT_FORMATS:
            return self.parse_text_file(file_path, output_dir, lang, **kwargs)
        else:
            # For unsupported file types, try as PDF
            logging.warning(
                f"Warning: Unsupported file extension '{ext}', "
                f"attempting to parse as PDF"
            )
            return self.parse_pdf(file_path, output_dir, method, lang, **kwargs)

    def check_installation(self) -> bool:
        """
        Check if MinerU 2.0 is properly installed

        Returns:
            bool: True if installation is valid, False otherwise
        """
        try:
            # Prepare subprocess parameters to hide console window on Windows
            import platform

            subprocess_kwargs = {
                "capture_output": True,
                "text": True,
                "check": True,
                "encoding": "utf-8",
                "errors": "ignore",
            }

            # Hide console window on Windows
            if platform.system() == "Windows":
                subprocess_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            result = subprocess.run(["mineru", "--version"], **subprocess_kwargs)
            logging.debug(f"MinerU version: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logging.debug(
                "MinerU 2.0 is not properly installed. "
                "Please install it using: pip install -U 'mineru[core]'"
            )
            return False


class DoclingParser(Parser):
    """
    Docling document parsing utility class.

    Specialized in parsing Office documents and HTML files, converting the content
    into structured data and generating markdown and JSON output.
    """

    # Define Docling-specific formats
    HTML_FORMATS = {".html", ".htm", ".xhtml"}

    def __init__(self) -> None:
        """Initialize DoclingParser"""
        super().__init__()

    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[str] = None,
        method: str = "auto",
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Parse PDF document using Docling

        Args:
            pdf_path: Path to the PDF file
            output_dir: Output directory path
            method: Parsing method (auto, txt, ocr)
            lang: Document language for OCR optimization
            **kwargs: Additional parameters for docling command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        try:
            # Convert to Path object for easier handling
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

            name_without_suff = pdf_path.stem

            # Prepare output directory
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = pdf_path.parent / "docling_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)

            # Run docling command
            self._run_docling_command(
                input_path=pdf_path,
                output_dir=base_output_dir,
                file_stem=name_without_suff,
                **kwargs,
            )

            # Read the generated output files
            content_list, _ = self._read_output_files(
                base_output_dir, name_without_suff
            )
            return content_list

        except Exception as e:
            logging.error(f"Error in parse_pdf: {str(e)}")
            raise

    def parse_document(
        self,
        file_path: Union[str, Path],
        method: str = "auto",
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Parse document using Docling based on file extension

        Args:
            file_path: Path to the file to be parsed
            method: Parsing method
            output_dir: Output directory path
            lang: Document language for optimization
            **kwargs: Additional parameters for docling command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        # Convert to Path object
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        # Get file extension
        ext = file_path.suffix.lower()

        # Choose appropriate parser based on file type
        if ext == ".pdf":
            return self.parse_pdf(file_path, output_dir, method, lang, **kwargs)
        elif ext in self.OFFICE_FORMATS:
            return self.parse_office_doc(file_path, output_dir, lang, **kwargs)
        elif ext in self.HTML_FORMATS:
            return self.parse_html(file_path, output_dir, lang, **kwargs)
        else:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Docling only supports PDF files, Office formats ({', '.join(self.OFFICE_FORMATS)}) "
                f"and HTML formats ({', '.join(self.HTML_FORMATS)})"
            )

    def _run_docling_command(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        file_stem: str,
        **kwargs,
    ) -> None:
        """
        Run docling command line tool

        Args:
            input_path: Path to input file or directory
            output_dir: Output directory path
            file_stem: File stem for creating subdirectory
            **kwargs: Additional parameters for docling command
        """
        # Create subdirectory structure similar to MinerU
        file_output_dir = Path(output_dir) / file_stem / "docling"
        file_output_dir.mkdir(parents=True, exist_ok=True)

        cmd_json = [
            "docling",
            "--output",
            str(file_output_dir),
            "--to",
            "json",
            str(input_path),
        ]
        cmd_md = [
            "docling",
            "--output",
            str(file_output_dir),
            "--to",
            "md",
            str(input_path),
        ]

        try:
            # Prepare subprocess parameters to hide console window on Windows
            import platform

            docling_subprocess_kwargs = {
                "capture_output": True,
                "text": True,
                "check": True,
                "encoding": "utf-8",
                "errors": "ignore",
            }

            # Hide console window on Windows
            if platform.system() == "Windows":
                docling_subprocess_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            result_json = subprocess.run(cmd_json, **docling_subprocess_kwargs)
            result_md = subprocess.run(cmd_md, **docling_subprocess_kwargs)
            logging.info("Docling command executed successfully")
            if result_json.stdout:
                logging.debug(f"JSON cmd output: {result_json.stdout}")
            if result_md.stdout:
                logging.debug(f"Markdown cmd output: {result_md.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running docling command: {e}")
            if e.stderr:
                logging.error(f"Error details: {e.stderr}")
            raise
        except FileNotFoundError:
            raise RuntimeError(
                "docling command not found. Please ensure Docling is properly installed."
            )

    def _read_output_files(
        self,
        output_dir: Path,
        file_stem: str,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Read the output files generated by docling and convert to MinerU format

        Args:
            output_dir: Output directory
            file_stem: File name without extension

        Returns:
            Tuple containing (content list JSON, Markdown text)
        """
        # Use subdirectory structure similar to MinerU
        file_subdir = output_dir / file_stem / "docling"
        md_file = file_subdir / f"{file_stem}.md"
        json_file = file_subdir / f"{file_stem}.json"

        # Read markdown content
        md_content = ""
        if md_file.exists():
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    md_content = f.read()
            except Exception as e:
                logging.warning(f"Could not read markdown file {md_file}: {e}")

        # Read JSON content and convert format
        content_list = []
        if json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    docling_content = json.load(f)
                    # Convert docling format to minerU format
                    content_list = self.read_from_block_recursive(
                        docling_content["body"],
                        "body",
                        file_subdir,
                        0,
                        "0",
                        docling_content,
                    )
            except Exception as e:
                logging.warning(f"Could not read or convert JSON file {json_file}: {e}")
        return content_list, md_content

    def read_from_block_recursive(
        self,
        block,
        type: str,
        output_dir: Path,
        cnt: int,
        num: str,
        docling_content: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        content_list = []
        if not block.get("children"):
            cnt += 1
            content_list.append(self.read_from_block(block, type, output_dir, cnt, num))
        else:
            if type not in ["groups", "body"]:
                cnt += 1
                content_list.append(
                    self.read_from_block(block, type, output_dir, cnt, num)
                )
            members = block["children"]
            for member in members:
                cnt += 1
                member_tag = member["$ref"]
                member_type = member_tag.split("/")[1]
                member_num = member_tag.split("/")[2]
                member_block = docling_content[member_type][int(member_num)]
                content_list.extend(
                    self.read_from_block_recursive(
                        member_block,
                        member_type,
                        output_dir,
                        cnt,
                        member_num,
                        docling_content,
                    )
                )
        return content_list

    def read_from_block(
        self, block, type: str, output_dir: Path, cnt: int, num: str
    ) -> Dict[str, Any]:
        if type == "texts":
            if block["label"] == "formula":
                return {
                    "type": "equation",
                    "img_path": "",
                    "text": block["orig"],
                    "text_format": "unkown",
                    "page_idx": cnt // 10,
                }
            else:
                return {
                    "type": "text",
                    "text": block["orig"],
                    "page_idx": cnt // 10,
                }
        elif type == "pictures":
            try:
                base64_uri = block["image"]["uri"]
                base64_str = base64_uri.split(",")[1]
                # Create images directory within the docling subdirectory
                image_dir = output_dir / "images"
                image_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                image_path = image_dir / f"image_{num}.png"
                with open(image_path, "wb") as f:
                    f.write(base64.b64decode(base64_str))
                return {
                    "type": "image",
                    "img_path": str(image_path.resolve()),  # Convert to absolute path
                    "image_caption": block.get("caption", ""),
                    "image_footnote": block.get("footnote", ""),
                    "page_idx": cnt // 10,
                }
            except Exception as e:
                logging.warning(f"Failed to process image {num}: {e}")
                return {
                    "type": "text",
                    "text": f"[Image processing failed: {block.get('caption', '')}]",
                    "page_idx": cnt // 10,
                }
        else:
            try:
                return {
                    "type": "table",
                    "img_path": "",
                    "table_caption": block.get("caption", ""),
                    "table_footnote": block.get("footnote", ""),
                    "table_body": block.get("data", []),
                    "page_idx": cnt // 10,
                }
            except Exception as e:
                logging.warning(f"Failed to process table {num}: {e}")
                return {
                    "type": "text",
                    "text": f"[Table processing failed: {block.get('caption', '')}]",
                    "page_idx": cnt // 10,
                }

    def parse_office_doc(
        self,
        doc_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Parse office document directly using Docling

        Supported formats: .doc, .docx, .ppt, .pptx, .xls, .xlsx

        Args:
            doc_path: Path to the document file
            output_dir: Output directory path
            lang: Document language for optimization
            **kwargs: Additional parameters for docling command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        try:
            # Convert to Path object
            doc_path = Path(doc_path)
            if not doc_path.exists():
                raise FileNotFoundError(f"Document file does not exist: {doc_path}")

            if doc_path.suffix.lower() not in self.OFFICE_FORMATS:
                raise ValueError(f"Unsupported office format: {doc_path.suffix}")

            name_without_suff = doc_path.stem

            # Prepare output directory
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = doc_path.parent / "docling_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)

            # Run docling command
            self._run_docling_command(
                input_path=doc_path,
                output_dir=base_output_dir,
                file_stem=name_without_suff,
                **kwargs,
            )

            # Read the generated output files
            content_list, _ = self._read_output_files(
                base_output_dir, name_without_suff
            )
            return content_list

        except Exception as e:
            logging.error(f"Error in parse_office_doc: {str(e)}")
            raise

    def parse_html(
        self,
        html_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Parse HTML document using Docling

        Supported formats: .html, .htm, .xhtml

        Args:
            html_path: Path to the HTML file
            output_dir: Output directory path
            lang: Document language for optimization
            **kwargs: Additional parameters for docling command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        try:
            # Convert to Path object
            html_path = Path(html_path)
            if not html_path.exists():
                raise FileNotFoundError(f"HTML file does not exist: {html_path}")

            if html_path.suffix.lower() not in self.HTML_FORMATS:
                raise ValueError(f"Unsupported HTML format: {html_path.suffix}")

            name_without_suff = html_path.stem

            # Prepare output directory
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = html_path.parent / "docling_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)

            # Run docling command
            self._run_docling_command(
                input_path=html_path,
                output_dir=base_output_dir,
                file_stem=name_without_suff,
                **kwargs,
            )

            # Read the generated output files
            content_list, _ = self._read_output_files(
                base_output_dir, name_without_suff
            )
            return content_list

        except Exception as e:
            logging.error(f"Error in parse_html: {str(e)}")
            raise

    def check_installation(self) -> bool:
        """
        Check if Docling is properly installed

        Returns:
            bool: True if installation is valid, False otherwise
        """
        try:
            # Prepare subprocess parameters to hide console window on Windows
            import platform

            subprocess_kwargs = {
                "capture_output": True,
                "text": True,
                "check": True,
                "encoding": "utf-8",
                "errors": "ignore",
            }

            # Hide console window on Windows
            if platform.system() == "Windows":
                subprocess_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            result = subprocess.run(["docling", "--version"], **subprocess_kwargs)
            logging.debug(f"Docling version: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logging.debug(
                "Docling is not properly installed. "
                "Please ensure it is installed correctly."
            )
            return False


class TianshuParser(Parser):
    """
    Tianshu remote document parsing service adapter.

    Parses documents by calling Tianshu HTTP API, which uses MinerU internally.
    Uses the /api/v1/tasks/{task_id}/data endpoint to retrieve complete content_list
    with full metadata (page_idx, bbox, caption, etc.).

    Attributes:
        tianshu_url: Tianshu service URL
        poll_interval: Polling interval in seconds
        timeout: Task timeout in seconds
        upload_images: Whether to upload images to MinIO
        session: HTTP session for connection reuse
    """

    def __init__(
        self,
        tianshu_url: str = "http://localhost:8000",
        poll_interval: float = 2.0,
        timeout: int = 3600,
        upload_images: bool = False,
    ) -> None:
        """
        Initialize TianshuParser.

        Args:
            tianshu_url: Tianshu service URL (e.g., http://localhost:8000)
            poll_interval: Polling interval for task status check (seconds)
            timeout: Maximum time to wait for task completion (seconds)
            upload_images: Whether to upload images to MinIO and return URLs
        """
        super().__init__()
        self.tianshu_url = tianshu_url.rstrip("/")
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.upload_images = upload_images
        self._session = None

    @property
    def session(self):
        """Lazy initialization of HTTP session for connection reuse."""
        if self._session is None:
            try:
                import requests
                self._session = requests.Session()
                # Set default timeout and retry strategy
                from requests.adapters import HTTPAdapter
                from urllib3.util.retry import Retry
                retry_strategy = Retry(
                    total=3,
                    backoff_factor=1,
                    status_forcelist=[500, 502, 503, 504],
                )
                adapter = HTTPAdapter(max_retries=retry_strategy)
                self._session.mount("http://", adapter)
                self._session.mount("https://", adapter)
            except ImportError:
                raise RuntimeError(
                    "requests library is required for TianshuParser. "
                    "Please install it using: pip install requests"
                )
        return self._session

    def _submit_task(
        self,
        file_path: Path,
        method: str = "auto",
        lang: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Submit a document parsing task to Tianshu service.

        Args:
            file_path: Path to the document file
            method: Parsing method (auto, txt, ocr)
            lang: Document language for OCR optimization
            **kwargs: Additional options (backend, formula_enable, table_enable, priority)

        Returns:
            str: Task ID returned by Tianshu service

        Raises:
            RuntimeError: If task submission fails
        """
        try:
            with open(file_path, "rb") as f:
                files = {"file": (file_path.name, f)}
                data = {
                    "lang": lang or "ch",
                    "backend": kwargs.get("backend", "pipeline"),
                    "method": method,
                    "formula_enable": kwargs.get("formula", True),
                    "table_enable": kwargs.get("table", True),
                    "priority": kwargs.get("priority", 0),
                }

                response = self.session.post(
                    f"{self.tianshu_url}/api/v1/tasks/submit",
                    files=files,
                    data=data,
                    timeout=60,
                )
                response.raise_for_status()

                result = response.json()
                if not result.get("success"):
                    raise RuntimeError(f"Task submission failed: {result}")

                task_id = result.get("task_id")
                if not task_id:
                    raise RuntimeError("No task_id returned from Tianshu")

                logging.info(f"[Tianshu] Task submitted: {task_id} for {file_path.name}")
                return task_id

        except Exception as e:
            logging.error(f"[Tianshu] Failed to submit task: {e}")
            raise RuntimeError(f"Tianshu task submission failed: {e}") from e

    def _poll_task(self, task_id: str) -> str:
        """
        Poll task status until completion.

        Args:
            task_id: Task ID to poll

        Returns:
            str: Final task status ('completed')

        Raises:
            TimeoutError: If task exceeds timeout
            RuntimeError: If task fails
        """
        start_time = time.time()
        last_status = None

        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                raise TimeoutError(
                    f"[Tianshu] Task {task_id} timed out after {self.timeout}s"
                )

            try:
                response = self.session.get(
                    f"{self.tianshu_url}/api/v1/tasks/{task_id}",
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()

                status = result.get("status")

                # Log status change
                if status != last_status:
                    logging.info(
                        f"[Tianshu] Task {task_id} status: {status} "
                        f"(elapsed: {elapsed:.1f}s)"
                    )
                    last_status = status

                if status == "completed":
                    return status

                elif status == "failed":
                    error_msg = result.get("error_message", "Unknown error")
                    raise RuntimeError(f"[Tianshu] Task failed: {error_msg}")

                elif status in ["pending", "processing"]:
                    time.sleep(self.poll_interval)

                else:
                    raise RuntimeError(f"[Tianshu] Unknown task status: {status}")

            except Exception as e:
                if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                    logging.warning(f"[Tianshu] Poll request timed out, retrying...")
                    time.sleep(self.poll_interval)
                else:
                    raise

    def _get_content_list(
        self, task_id: str, output_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Get content_list from Tianshu using the /data endpoint.

        Args:
            task_id: Task ID
            output_dir: Optional output directory to save content_list.json

        Returns:
            List[Dict[str, Any]]: Content list in MinerU format

        Raises:
            RuntimeError: If content_list retrieval fails
        """
        try:
            response = self.session.get(
                f"{self.tianshu_url}/api/v1/tasks/{task_id}/data",
                params={
                    "include_fields": "content_list,images,md",
                    "upload_images": self.upload_images,
                },
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()

            if not result.get("success"):
                raise RuntimeError(f"Failed to get content_list: {result}")

            # Extract content_list from response
            # Response format: {"data": {"content_list": {"content": [...], "file_name": "..."}, "images": [...]}}
            data = result.get("data", {})
            content_list_wrapper = data.get("content_list", {})

            if isinstance(content_list_wrapper, dict):
                content_list = content_list_wrapper.get("content", [])
            else:
                content_list = content_list_wrapper or []

            if not content_list:
                logging.warning(f"[Tianshu] Empty content_list returned for task {task_id}")

            # Merge image URLs from images field into content_list (when upload_images=True)
            logging.info(f"[Tianshu] upload_images={self.upload_images}")
            if self.upload_images:
                images_data = data.get("images", [])
                logging.info(f"[Tianshu] images_data count: {len(images_data) if images_data else 0}")
                content_list = self._merge_image_urls(content_list, images_data)
            else:
                logging.info("[Tianshu] upload_images is False, skipping URL merge")

            logging.info(
                f"[Tianshu] Retrieved content_list with {len(content_list)} items"
            )

            # Optionally save to output directory
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                json_path = output_dir / f"{task_id}_content_list.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(content_list, f, ensure_ascii=False, indent=2)
                logging.info(f"[Tianshu] Saved content_list to: {json_path}")

            return content_list

        except Exception as e:
            logging.error(f"[Tianshu] Failed to get content_list: {e}")
            raise RuntimeError(f"Failed to get content_list from Tianshu: {e}") from e

    def _merge_image_urls(
        self, content_list: List[Dict[str, Any]], images_data: Any
    ) -> List[Dict[str, Any]]:
        """
        Merge MinIO image URLs from images field into content_list items.

        When upload_images=True, Tianshu uploads images to MinIO and returns URLs
        in the images field. This method maps those URLs back to the corresponding
        items in content_list using the path field as the key.

        Args:
            content_list: Content list with img_path fields (relative paths)
            images_data: Images data from Tianshu response. Can be:
                - A dict with "content" key containing list of image info
                - A list of image info dicts, each containing:
                    - name: Image filename
                    - path: Relative path (matches img_path in content_list)
                    - url: MinIO URL for the image

        Returns:
            Updated content_list with url fields added to image items
        """
        if not images_data:
            logging.info("[Tianshu] No images_data provided")
            return content_list

        # Handle different response formats
        # Format 1: {"count": N, "list": [...]} (Tianshu format)
        # Format 2: {"content": [...], "file_name": "..."} (wrapper format)
        # Format 3: [...] (direct list)
        if isinstance(images_data, dict):
            # Try "list" key first (Tianshu format)
            if "list" in images_data:
                images_list = images_data.get("list", [])
                logging.info(f"[Tianshu] images_data is dict, extracted {len(images_list)} items from 'list'")
            else:
                images_list = images_data.get("content", [])
                logging.info(f"[Tianshu] images_data is dict, extracted {len(images_list)} items from 'content'")
        elif isinstance(images_data, list):
            images_list = images_data
            logging.info(f"[Tianshu] images_data is list with {len(images_list)} items")
        else:
            logging.warning(f"[Tianshu] Unexpected images_data type: {type(images_data)}")
            return content_list

        if not images_list:
            logging.info("[Tianshu] images_list is empty")
            return content_list

        # Log first item to debug format
        if images_list:
            first_item = images_list[0]
            logging.info(f"[Tianshu] First image item type: {type(first_item)}, value: {first_item}")

        # Build path -> url mapping
        # Expected format: [{"name": "xxx.jpg", "path": "images/xxx.jpg", "url": "http://..."}]
        path_to_url = {}
        for img in images_list:
            if isinstance(img, dict):
                path = img.get("path")
                url = img.get("url")
                if path and url:
                    path_to_url[path] = url
            else:
                logging.warning(f"[Tianshu] Skipping non-dict image item: {type(img)}")

        if not path_to_url:
            logging.warning("[Tianshu] No valid image URLs found in images data")
            return content_list

        logging.info(f"[Tianshu] Built path->url mapping for {len(path_to_url)} images")

        # Merge URLs into content_list
        merged_count = 0
        for item in content_list:
            # Check for image items (type == "image" or has img_path)
            if item.get("type") == "image" or "img_path" in item:
                img_path = item.get("img_path")
                if img_path and img_path in path_to_url:
                    item["url"] = path_to_url[img_path]
                    merged_count += 1
                    logging.debug(f"[Tianshu] Merged URL for: {img_path}")

        logging.info(f"[Tianshu] Merged {merged_count} image URLs into content_list")

        return content_list

    def _download_images(
        self, task_id: str, content_list: List[Dict[str, Any]], output_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Download images from Tianshu server and update img_path in content_list.

        Args:
            task_id: Tianshu task ID
            content_list: Content list with relative image paths
            output_dir: Local output directory

        Returns:
            Updated content_list with absolute local image paths
        """
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        downloaded_count = 0
        failed_count = 0

        for item in content_list:
            # Check for image items
            if item.get("type") == "image" or "img_path" in item:
                img_path = item.get("img_path", "")
                if not img_path:
                    continue

                # Skip if already absolute path or URL
                if img_path.startswith("/") or img_path.startswith("http"):
                    continue

                # Extract image filename
                img_filename = Path(img_path).name

                try:
                    # Download image from Tianshu
                    response = self.session.get(
                        f"{self.tianshu_url}/api/v1/tasks/{task_id}/images/{img_filename}",
                        timeout=30,
                    )

                    if response.status_code == 200:
                        # Save image locally
                        local_img_path = images_dir / img_filename
                        with open(local_img_path, "wb") as f:
                            f.write(response.content)

                        # Update path in content_list to absolute path
                        item["img_path"] = str(local_img_path.absolute())
                        downloaded_count += 1
                        logging.debug(f"[Tianshu] Downloaded image: {img_filename}")
                    else:
                        logging.warning(
                            f"[Tianshu] Failed to download image {img_filename}: "
                            f"HTTP {response.status_code}"
                        )
                        failed_count += 1

                except Exception as e:
                    logging.warning(f"[Tianshu] Error downloading image {img_filename}: {e}")
                    failed_count += 1

        if downloaded_count > 0 or failed_count > 0:
            logging.info(
                f"[Tianshu] Image download complete: "
                f"{downloaded_count} succeeded, {failed_count} failed"
            )

        return content_list

    # Text file formats that can be read directly without Tianshu parsing
    TEXT_FORMATS = {".md", ".txt", ".markdown", ".rst", ".text"}

    def parse_document(
        self,
        file_path: Union[str, Path],
        method: str = "auto",
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Parse document using Tianshu remote service.

        For text-based files (.md, .txt, etc.), reads content directly without
        calling Tianshu, as these files don't need parsing/OCR.

        Args:
            file_path: Path to the document file
            method: Parsing method (auto, txt, ocr)
            output_dir: Output directory path
            lang: Document language for OCR optimization
            **kwargs: Additional parameters (backend, formula, table, priority)

        Returns:
            List[Dict[str, Any]]: List of content blocks in MinerU format
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        # For text-based files, read directly without Tianshu
        ext = file_path.suffix.lower()
        if ext in self.TEXT_FORMATS:
            logging.info(f"[Tianshu] Text file detected, reading directly: {file_path.name}")
            return self._parse_text_file(file_path)

        logging.info(f"[Tianshu] Parsing document: {file_path.name}")

        # Step 1: Submit task
        task_id = self._submit_task(file_path, method, lang, **kwargs)

        # Step 2: Poll until completion
        self._poll_task(task_id)

        # Step 3: Get content_list
        output_path = Path(output_dir) if output_dir else Path("output")
        content_list = self._get_content_list(task_id, output_path)

        # Step 4: Download images from Tianshu server (if not using MinIO upload)
        if not self.upload_images and output_path:
            content_list = self._download_images(task_id, content_list, output_path)

        logging.info(
            f"[Tianshu] Successfully parsed {file_path.name}: "
            f"{len(content_list)} content blocks"
        )

        return content_list

    def _parse_text_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Parse text-based file (Markdown, TXT, etc.) directly without Tianshu.

        These files are already in readable format and don't need OCR or parsing.
        Simply reads the content and converts to content_list format.

        Args:
            file_path: Path to the text file

        Returns:
            List[Dict[str, Any]]: Content list with text blocks
        """
        try:
            # Try common encodings
            content = None
            for encoding in ["utf-8", "gbk", "gb2312", "latin-1"]:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                raise ValueError(f"Failed to decode file with supported encodings: {file_path}")

            if not content.strip():
                logging.warning(f"[Tianshu] Text file is empty: {file_path.name}")
                return []

            # Build content_list from text content
            # Split by paragraphs (double newlines) to create separate text blocks
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

            content_list = []
            for i, paragraph in enumerate(paragraphs):
                content_list.append({
                    "type": "text",
                    "text": paragraph,
                    "page_idx": 0,  # Text files don't have pages
                })

            logging.info(
                f"[Tianshu] Successfully read text file {file_path.name}: "
                f"{len(content_list)} text blocks, {len(content)} characters"
            )

            return content_list

        except Exception as e:
            logging.error(f"[Tianshu] Failed to read text file {file_path}: {e}")
            raise

    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[str] = None,
        method: str = "auto",
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Parse PDF document using Tianshu.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Output directory path
            method: Parsing method (auto, txt, ocr)
            lang: Document language for OCR optimization
            **kwargs: Additional parameters for Tianshu

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        return self.parse_document(pdf_path, method, output_dir, lang, **kwargs)

    def parse_image(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Parse image document using Tianshu.

        Args:
            image_path: Path to the image file
            output_dir: Output directory path
            lang: Document language for OCR optimization
            **kwargs: Additional parameters for Tianshu

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        return self.parse_document(image_path, "ocr", output_dir, lang, **kwargs)

    def parse_office_doc(
        self,
        doc_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Parse Office document using Tianshu.

        This method first converts the Office document to PDF locally using LibreOffice,
        then sends the PDF to Tianshu for parsing. This approach ensures:
        1. Complete content_list with full metadata (page_idx, bbox, etc.)
        2. Consistency with local MinerU parsing results
        3. No information loss from MarkItDown conversion

        Supported formats: .doc, .docx, .ppt, .pptx, .xls, .xlsx

        Args:
            doc_path: Path to the Office document file
            output_dir: Output directory path
            lang: Document language for OCR optimization
            **kwargs: Additional parameters for Tianshu

        Returns:
            List[Dict[str, Any]]: List of content blocks

        Raises:
            ValueError: If the file format is not supported
            RuntimeError: If LibreOffice conversion fails
        """
        doc_path = Path(doc_path)
        if doc_path.suffix.lower() not in self.OFFICE_FORMATS:
            raise ValueError(
                f"Unsupported office format: {doc_path.suffix}. "
                f"Supported formats: {', '.join(self.OFFICE_FORMATS)}"
            )

        logging.info(
            f"[Tianshu] Converting Office document to PDF locally: {doc_path.name}"
        )

        try:
            # Convert Office document to PDF using base class method (requires LibreOffice)
            pdf_path = self.convert_office_to_pdf(doc_path, output_dir)
            logging.info(f"[Tianshu] Converted to PDF: {pdf_path}")

            # Parse the converted PDF using Tianshu
            return self.parse_document(pdf_path, "auto", output_dir, lang, **kwargs)

        except Exception as e:
            logging.error(f"[Tianshu] Failed to process Office document: {e}")
            raise

    def check_installation(self) -> bool:
        """
        Check if Tianshu service is available.

        Returns:
            bool: True if Tianshu service is healthy, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.tianshu_url}/api/v1/health",
                timeout=10,
            )
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "healthy":
                    logging.info(f"[Tianshu] Service is healthy at {self.tianshu_url}")
                    return True
            logging.warning(f"[Tianshu] Service unhealthy: {response.text}")
            return False
        except Exception as e:
            logging.warning(f"[Tianshu] Service unavailable at {self.tianshu_url}: {e}")
            return False

    def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            self._session.close()
            self._session = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


def main():
    """
    Main function to run the document parser from command line
    """
    parser = argparse.ArgumentParser(
        description="Parse documents using MinerU 2.0 or Docling"
    )
    parser.add_argument("file_path", help="Path to the document to parse")
    parser.add_argument("--output", "-o", help="Output directory path")
    parser.add_argument(
        "--method",
        "-m",
        choices=["auto", "txt", "ocr"],
        default="auto",
        help="Parsing method (auto, txt, ocr)",
    )
    parser.add_argument(
        "--lang",
        "-l",
        help="Document language for OCR optimization (e.g., ch, en, ja)",
    )
    parser.add_argument(
        "--backend",
        "-b",
        choices=[
            "pipeline",
            "vlm-transformers",
            "vlm-sglang-engine",
            "vlm-sglang-client",
        ],
        default="pipeline",
        help="Parsing backend",
    )
    parser.add_argument(
        "--device",
        "-d",
        help="Inference device (e.g., cpu, cuda, cuda:0, npu, mps)",
    )
    parser.add_argument(
        "--source",
        choices=["huggingface", "modelscope", "local"],
        default="huggingface",
        help="Model source",
    )
    parser.add_argument(
        "--no-formula",
        action="store_true",
        help="Disable formula parsing",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="Disable table parsing",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Display content statistics"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check parser installation",
    )
    parser.add_argument(
        "--parser",
        choices=["mineru", "docling"],
        default="mineru",
        help="Parser selection",
    )
    parser.add_argument(
        "--vlm_url",
        help="When the backend is `vlm-sglang-client`, you need to specify the server_url, for example:`http://127.0.0.1:30000`",
    )

    args = parser.parse_args()

    # Check installation if requested
    if args.check:
        doc_parser = DoclingParser() if args.parser == "docling" else MineruParser()
        if doc_parser.check_installation():
            print(f"✅ {args.parser.title()} is properly installed")
            return 0
        else:
            print(f"❌ {args.parser.title()} installation check failed")
            return 1

    try:
        # Parse the document
        doc_parser = DoclingParser() if args.parser == "docling" else MineruParser()
        content_list = doc_parser.parse_document(
            file_path=args.file_path,
            method=args.method,
            output_dir=args.output,
            lang=args.lang,
            backend=args.backend,
            device=args.device,
            source=args.source,
            formula=not args.no_formula,
            table=not args.no_table,
            vlm_url=args.vlm_url,
        )

        print(f"✅ Successfully parsed: {args.file_path}")
        print(f"📊 Extracted {len(content_list)} content blocks")

        # Display statistics if requested
        if args.stats:
            print("\n📈 Document Statistics:")
            print(f"Total content blocks: {len(content_list)}")

            # Count different types of content
            content_types = {}
            for item in content_list:
                if isinstance(item, dict):
                    content_type = item.get("type", "unknown")
                    content_types[content_type] = content_types.get(content_type, 0) + 1

            if content_types:
                print("\n📋 Content Type Distribution:")
                for content_type, count in sorted(content_types.items()):
                    print(f"  • {content_type}: {count}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
