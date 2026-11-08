"""
Unit tests for Hinglish text preprocessing module (src.preprocessing).
"""

import pytest
from src.preprocessing import clean_text, preprocess_text


class TestHinglishPreprocessing:
    """Comprehensive test suite for Hinglish text cleaning and preprocessing."""

    def test_normal_hinglish_sentences(self):
        """Validates that regular Hinglish phrases and code-mixed vocabulary remain intact."""
        sample = "Yeh banda bilkul bakwaas baatein kar raha hai."
        assert clean_text(sample) == sample
        assert preprocess_text(sample) == sample

        mixed = "Bhai please stop doing this, dimag kharab mat karo."
        assert clean_text(mixed) == mixed

    def test_empty_none_and_invalid_inputs(self):
        """Validates safe handling for None, empty strings, whitespace, and non-string inputs."""
        assert clean_text(None) == ""
        assert clean_text("") == ""
        assert clean_text("   \t  \n  ") == ""
        assert clean_text(12345) == "12345"
        assert clean_text(3.14) == "3.14"
        assert preprocess_text(None) == ""

    def test_url_removal(self):
        """Validates removal of various URL formats without affecting other text."""
        input_text = "Check this out https://example.com/post?id=123 bahut bura lag raha hai"
        expected = "Check this out bahut bura lag raha hai"
        assert clean_text(input_text) == expected

        input_www = "Ye video dekho www.youtube.com/watch?v=xyz pagalpan hai"
        expected_www = "Ye video dekho pagalpan hai"
        assert clean_text(input_www) == expected_www

        only_url = "https://t.co/abc12345"
        assert clean_text(only_url) == ""

    def test_html_tag_and_entity_removal(self):
        """Validates cleaning of HTML tags and unescaping of entities."""
        html_input = "<p>Yeh <b>kya</b> ho raha hai? &amp; kyu ho raha hai?</p>"
        expected = "Yeh kya ho raha hai? & kyu ho raha hai?"
        assert clean_text(html_input) == expected

    def test_extra_whitespace_normalization(self):
        """Validates reduction of tabs, multiple spaces, and excess newlines."""
        input_text = "  Arey    bhai,    kya    scene   hai?  \n\n\n  Sab theek?  "
        expected = "Arey bhai, kya scene hai? \n Sab theek?"
        assert clean_text(input_text) == expected

    def test_repeated_characters_careful_handling(self):
        """Validates that runs of 3+ chars are reduced to 2, keeping legitimate double letters intact."""
        # 3+ characters reduced to 2
        assert clean_text("bahuuuuut bakwaaaas hai ye") == "bahuut bakwaas hai ye"
        assert clean_text("kuttaaaaa kameenaaaa") == "kuttaa kameenaa"
        assert clean_text("sooooo goooood") == "soo good"

        # Legitimate 2-character sequences in Hinglish must be preserved
        assert clean_text("haan") == "haan"
        assert clean_text("bhaii") == "bhaii"
        assert clean_text("kall") == "kall"
        assert clean_text("too") == "too"

    def test_emojis_and_punctuation_preservation(self):
        """Validates that emojis, question marks, and exclamation marks are retained."""
        input_text = "Tu pagal hai kya??? 😡🤬 Stop it now!!!"
        # Repeated punctuation '???' and '!!!' gets reduced to '??' and '!!' safely
        expected = "Tu pagal hai kya?? 😡🤬 Stop it now!!"
        assert clean_text(input_text) == expected

        emoji_only = "❤️ 🔥 💯"
        assert clean_text(emoji_only) == "❤️ 🔥 💯"

    def test_negation_words_preservation(self):
        """Validates that Hindi/Hinglish and English negations are never removed or altered."""
        sentences = [
            "Mujhe ye bilkul pasand nahi hai",
            "Wahan mat jao",
            "Kabhi nahi sudhrega ye banda",
            "This is not acceptable, na karo aisi harkat",
        ]
        for s in sentences:
            assert clean_text(s) == s

    def test_unicode_normalization(self):
        """Validates NFKC Unicode standardization for full-width or accented characters."""
        # Full-width Latin letters: Ｈｅｌｌｏ
        styled_text = "\uff28\uff45\uff4c\uff4c\uff4f bhai"
        assert clean_text(styled_text) == "Hello bhai"

    def test_preserves_critical_signals(self):
        """Validates that offensive / hate terms are not stripped, as they are crucial for classification."""
        toxic_sample = "kutta kamina kahin ka"
        assert clean_text(toxic_sample) == toxic_sample
