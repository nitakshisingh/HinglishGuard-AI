"""
Hinglish Text Preprocessing Module for HinglishGuard AI.

Provides specialized cleaning, Unicode normalization, URL/HTML tag removal,
repeated character moderation, and whitespace normalization for code-mixed
Hindi-English (Hinglish) and Latin/Devanagari text.
"""

import html
import re
import unicodedata
from typing import Any, Optional

# Regex for URLs (http, https, ftp, and www prefixed links)
URL_PATTERN = re.compile(
    r"(?:https?://|www\.)[^\s/$.?#].[^\s]*",
    re.IGNORECASE,
)

# Regex for HTML tags
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

# Regex for repeated characters (3 or more consecutive identical characters -> reduced to 2)
# e.g., 'bahhhhuuut' -> 'bahhuut', 'kuttaaaaa' -> 'kuttaa'
REPEATED_CHARS_PATTERN = re.compile(r"(.)\1{2,}", re.IGNORECASE)

# Regex for multi-whitespace normalization
WHITESPACE_PATTERN = re.compile(r"[^\S\r\n]+")

# Regex for multiple consecutive newlines
NEWLINES_PATTERN = re.compile(r"\n\s*\n+")


def clean_text(text: Any) -> str:
    """Clean and normalize raw Hinglish text.

    Key operations:
    1. Safe type handling: Handles None, non-string, or empty input cleanly.
    2. Unicode normalization: Uses NFKC form to unify composite and stylized characters.
    3. HTML entity unescaping and HTML tag removal.
    4. URL removal: Strips web links while maintaining surrounding content structure.
    5. Repeated characters reduction: Reduces runs of 3+ identical characters down to 2
       (preserving valid Hinglish phonetic spellings such as 'haan', 'bhaii', 'kuttaa').
    6. Whitespace cleanup: Normalizes irregular spaces and redundant line breaks.

    Preserves:
    - Hinglish words and vocabulary (no translation or phonetic distortion).
    - Negation terms ('nahi', 'mat', 'not', 'kabhi nahi', 'na').
    - Offensive/hate speech signals (needed for accurate downstream classification).
    - Emojis and meaningful punctuation marks (e.g., '?', '!', ':)', '😡').

    Args:
        text: Raw input text of any type.

    Returns:
        str: Cleaned and normalized text string. Returns empty string for invalid inputs.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    if not text.strip():
        return ""

    # 1. Unicode Normalization (NFKC standardizes compatibility characters and accents)
    cleaned = unicodedata.normalize("NFKC", text)

    # 2. Decode HTML entities (e.g., &amp; -> &, &lt; -> <)
    cleaned = html.unescape(cleaned)

    # 3. Strip HTML tags (e.g., <div>, <br/>, <a href="...">)
    cleaned = HTML_TAG_PATTERN.sub(" ", cleaned)

    # 4. Remove URLs
    cleaned = URL_PATTERN.sub("", cleaned)

    # 5. Handle excessive repeated characters (3+ identical characters condensed to 2)
    # Hinglish words often double letters for long vowels/gemination (e.g. 'haan', 'kuttaa', 'too'),
    # but elongated typing (e.g., 'bahuuuuut', 'bakwaaaaas') introduces noise.
    cleaned = REPEATED_CHARS_PATTERN.sub(r"\1\1", cleaned)

    # 6. Normalize horizontal whitespaces and excess newlines
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned)
    cleaned = NEWLINES_PATTERN.sub("\n", cleaned)

    return cleaned.strip()


def preprocess_text(text: Any) -> str:
    """Main preprocessing entrypoint for text classification pipelines.

    Passes the input through `clean_text` and prepares it for feature
    extraction or tokenization in baseline and transformer models.

    Args:
        text: Raw input text.

    Returns:
        str: Normalized and preprocessed text string.
    """
    return clean_text(text)
