import re
import argparse
from pathlib import Path
import subprocess
from functools import lru_cache
import unicodedata

import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageOps


KAN_RANGE_RE = re.compile(r"[\u0C80-\u0CFF]")  # Kannada Unicode block

def kannada_ratio(s: str) -> float:
    if not s:
        return 0.0
    kan = len(KAN_RANGE_RE.findall(s))
    return kan / max(1, len(s))

def extract_with_pymupdf(pdf_path: Path, page_index: int) -> str:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    text = page.get_text("text")
    doc.close()
    return text or ""

@lru_cache(maxsize=1)
def _tesseract_list_langs() -> set[str]:
    """
    Return the set of languages available to the `tesseract` CLI.
    Cached so we don't shell out on every page.
    """
    try:
        proc = subprocess.run(
            ["tesseract", "--list-langs"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "Tesseract is not installed or not on PATH. Install it (macOS): `brew install tesseract`"
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to run `tesseract --list-langs`.\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}"
        ) from e

    langs: set[str] = set()
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line or line.lower().startswith("list of available languages"):
            continue
        langs.add(line)
    return langs

def ensure_tesseract_lang(lang: str) -> None:
    langs = _tesseract_list_langs()
    if lang in langs:
        return

    # Keep this message very actionable (this is the error the user hit).
    raise RuntimeError(
        "Tesseract language data is missing.\n"
        f"- Requested lang: {lang}\n"
        f"- Available langs: {', '.join(sorted(langs)) if langs else '(none)'}\n\n"
        "Fix (macOS/Homebrew):\n"
        "  1) `brew install tesseract-lang`\n"
        "  2) Re-check: `tesseract --list-langs | grep -E '^kan$'`\n\n"
        "If `tesseract-lang` is not available in your Homebrew setup, install manually:\n"
        f"  - Download `kan.traineddata` into: {(Path('/opt/homebrew/share/tessdata'))}\n"
        "  - Then re-run `tesseract --list-langs`.\n"
    )

def ocr_page(
    pdf_path: Path,
    page_number_1based: int,
    dpi: int = 300,
    lang: str = "kan",
    psm: int = 6,
    oem: int = 1,
) -> str:
    # Convert just one page to an image
    images = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        first_page=page_number_1based,
        last_page=page_number_1based
    )
    img = images[0]

    # Light preprocessing helps a LOT for scanned books
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    # Optional: simple thresholding
    img = img.point(lambda x: 0 if x < 180 else 255, "1")

    # Kannada OCR
    ensure_tesseract_lang(lang)
    config = f"--oem {oem} --psm {psm}"
    return pytesseract.image_to_string(img, lang=lang, config=config) or ""

_CTRL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_SPACES_RE = re.compile(r"[ \t]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")
_PAGE_NUMBER_LINE_RE = re.compile(r"^\s*[\[\(]?\s*\d{1,5}\s*[\]\)]?\s*$")

def clean_text_basic(text: str) -> str:
    # Minimal normalization; keeps most structure intact.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL_CHARS_RE.sub(" ", text)
    text = _SPACES_RE.sub(" ", text)
    text = _BLANKLINES_RE.sub("\n\n", text)
    return text.strip()

def clean_text_train(text: str) -> str:
    """
    More aggressive cleaning aimed at tokenizer training:
    - Unicode normalize (NFC)
    - drop control chars / normalize whitespace
    - drop lines that look like page numbers
    - join wrapped lines into paragraphs while preserving blank lines as paragraph breaks
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00A0", " ")  # NBSP -> space
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL_CHARS_RE.sub("\n", text)

    # Normalize per-line whitespace early
    raw_lines = [ln.strip() for ln in text.split("\n")]

    filtered: list[str] = []
    for ln in raw_lines:
        if not ln:
            filtered.append("")
            continue
        # Remove standalone page-number lines (very common in PDF/OCR dumps)
        if _PAGE_NUMBER_LINE_RE.match(ln):
            continue
        # Collapse internal whitespace
        ln = _SPACES_RE.sub(" ", ln).strip()
        if ln:
            filtered.append(ln)

    # Join wrapped lines within paragraphs
    paras: list[str] = []
    buf: list[str] = []

    def flush():
        if not buf:
            return
        paras.append(" ".join(buf).strip())
        buf.clear()

    # Treat these as sentence/paragraph-ish line endings; if a line ends with one,
    # we keep the following line as a new buffer element (still joined with a space),
    # but the heuristic below is more conservative.
    strong_end = set(".!?।॥…:;\"”'’)")

    for ln in filtered:
        if ln == "":
            flush()
            continue

        if not buf:
            buf.append(ln)
            continue

        prev = buf[-1]

        # Dehyphenate ASCII word-breaks: "exam-\nple" -> "example"
        if prev.endswith("-") and re.search(r"[A-Za-z]-$", prev) and re.match(r"^[A-Za-z]", ln):
            buf[-1] = prev[:-1] + ln
            continue

        # If previous line clearly ends a sentence, keep as separate chunk (still within paragraph)
        if prev and prev[-1] in strong_end:
            buf.append(ln)
            continue

        # Otherwise assume it's wrapped prose: join with space
        buf.append(ln)

    flush()

    out = "\n\n".join(p for p in paras if p)
    out = _SPACES_RE.sub(" ", out)
    out = _BLANKLINES_RE.sub("\n\n", out)
    return out.strip()

def clean_text(text: str, mode: str = "train") -> str:
    if mode == "basic":
        return clean_text_basic(text)
    if mode == "train":
        return clean_text_train(text)
    raise ValueError(f"Unknown clean mode: {mode}")

def process_pdf(
    pdf_path: Path,
    out_txt: Path,
    max_pages: int,
    min_kan_ratio: float,
    force_ocr: bool,
    clean_mode: str,
    ocr_dpi: int,
    ocr_psm: int,
    ocr_oem: int,
):
    chunks = []
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    doc.close()

    pages_to_do = total_pages if max_pages <= 0 else min(max_pages, total_pages)
    if max_pages > 0 and pages_to_do < total_pages:
        print(f"[INFO] Sampling first {pages_to_do} pages (total pages: {total_pages}). Use `--pages 0` for all pages.")

    for i in range(pages_to_do):
        page_num_1based = i + 1

        if not force_ocr:
            raw = extract_with_pymupdf(pdf_path, i)
            ratio = kannada_ratio(raw)

            # Heuristic: if it's empty or not mostly Kannada, OCR it
            if raw.strip() and ratio >= min_kan_ratio:
                chunks.append(raw)
                print(f"[OK]  Page {page_num_1based}: text-extract (kan_ratio={ratio:.2f})")
                continue
            else:
                print(f"[FALLBACK] Page {page_num_1based}: OCR (kan_ratio={ratio:.2f}, len={len(raw)})")

        ocr = ocr_page(
            pdf_path=pdf_path,
            page_number_1based=page_num_1based,
            dpi=ocr_dpi,
            lang="kan",
            psm=ocr_psm,
            oem=ocr_oem,
        )
        chunks.append(ocr)
        print(f"[OCR] Page {page_num_1based}: got {len(ocr)} chars")

    final = clean_text("\n\n".join(chunks), mode=clean_mode)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(final, encoding="utf-8")
    print(f"\n✅ Wrote: {out_txt}  (pages processed: {pages_to_do}/{total_pages})")

def _default_pdf_dir() -> Path:
    """
    User asked for folder 'KanndaNovels' (typo), but repo contains 'KannadaNovels'.
    Support both; pick the first that exists.
    """
    candidates = [Path("KanndaNovels"), Path("KannadaNovels")]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    # Fall back to the (correct) name for clearer error messages
    return Path("KannadaNovels")

def process_pdf_dir(
    pdf_dir: Path,
    out_dir: Path,
    max_pages: int,
    min_kan_ratio: float,
    force_ocr: bool,
    clean_mode: str,
    ocr_dpi: int,
    ocr_psm: int,
    ocr_oem: int,
    recursive: bool = False,
):
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdfs = sorted(p for p in pdf_dir.glob(pattern) if p.is_file())
    if not pdfs:
        print(f"⚠️  No PDFs found in: {pdf_dir} (pattern={pattern})")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"📚 Found {len(pdfs)} PDFs in {pdf_dir}. Writing .txt files to {out_dir}\n")

    for idx, pdf_path in enumerate(pdfs, start=1):
        out_txt = out_dir / f"{pdf_path.stem}.txt"
        print(f"\n=== [{idx}/{len(pdfs)}] {pdf_path.name} ===")
        process_pdf(
            pdf_path=pdf_path,
            out_txt=out_txt,
            max_pages=max_pages,
            min_kan_ratio=min_kan_ratio,
            force_ocr=force_ocr,
            clean_mode=clean_mode,
            ocr_dpi=ocr_dpi,
            ocr_psm=ocr_psm,
            ocr_oem=ocr_oem,
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", help="Path to a single PDF (single-file mode)")
    ap.add_argument("--out", help="Output .txt path (single-file mode)")
    ap.add_argument("--pdf-dir", default=str(_default_pdf_dir()),
                    help="Directory containing PDFs (batch mode). Defaults to KannadaNovels (or KanndaNovels if present).")
    ap.add_argument("--out-dir", default="txt_out",
                    help="Directory to write .txt files to (batch mode). One .txt per PDF.")
    ap.add_argument("--recursive", action="store_true", help="In batch mode, search PDFs recursively under --pdf-dir")
    ap.add_argument("--pages", type=int, default=5,
                    help="How many pages to process. Use 5 for sampling; use 0 to process ALL pages.")
    ap.add_argument("--min-kan-ratio", type=float, default=0.15,
                    help="If extracted text has Kannada ratio below this, OCR is used")
    ap.add_argument("--force-ocr", action="store_true", help="OCR every page (useful for scanned/legacy-font PDFs)")
    ap.add_argument("--clean-mode", choices=["basic", "train"], default="train",
                    help="Text cleaning mode. 'train' is recommended for tokenizer training.")
    ap.add_argument("--ocr-dpi", type=int, default=300, help="DPI used for OCR rendering")
    ap.add_argument("--ocr-psm", type=int, default=6, help="Tesseract page segmentation mode (PSM)")
    ap.add_argument("--ocr-oem", type=int, default=1, help="Tesseract OCR engine mode (OEM)")
    args = ap.parse_args()

    # If user provides --pdf, run single-file mode (backward compatible).
    # Otherwise, default to batch mode reading all PDFs from --pdf-dir.
    if args.pdf:
        if not args.out:
            ap.error("--out is required when using --pdf (single-file mode).")
        process_pdf(
            pdf_path=Path(args.pdf),
            out_txt=Path(args.out),
            max_pages=args.pages,
            min_kan_ratio=args.min_kan_ratio,
            force_ocr=args.force_ocr,
            clean_mode=args.clean_mode,
            ocr_dpi=args.ocr_dpi,
            ocr_psm=args.ocr_psm,
            ocr_oem=args.ocr_oem,
        )
        return

    process_pdf_dir(
        pdf_dir=Path(args.pdf_dir),
        out_dir=Path(args.out_dir),
        max_pages=args.pages,
        min_kan_ratio=args.min_kan_ratio,
        force_ocr=args.force_ocr,
        clean_mode=args.clean_mode,
        ocr_dpi=args.ocr_dpi,
        ocr_psm=args.ocr_psm,
        ocr_oem=args.ocr_oem,
        recursive=args.recursive,
    )

if __name__ == "__main__":
    main()
