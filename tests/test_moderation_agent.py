"""
Unit tests for Gemini Moderation Agent in HinglishGuard AI.
Tests contextual moderation, structured JSON parsing, exception handling,
and safe non-fabricating fallbacks using mocks (no live API calls).
"""

from unittest.mock import MagicMock
import pytest

from src.moderation_agent import ModerationAgent, GeminiModerationResponse


@pytest.fixture
def mock_gemini_client():
    """Create a mock genai.Client instance."""
    mock_client = MagicMock()
    return mock_client


def test_agent_unconfigured_when_no_api_key():
    """Verify agent reports unavailable status when API key is None or placeholder."""
    agent = ModerationAgent(api_key="")
    assert not agent.is_configured
    result = agent.analyze_text("Test Hinglish text")
    assert not result["is_available"]
    assert result["status"] == "UNAVAILABLE"
    assert result["detected_category"] is None
    assert "not configured" in result["explanation"]


def test_agent_unconfigured_when_placeholder_api_key():
    """Verify agent detects default placeholder key as unconfigured."""
    agent = ModerationAgent(api_key="your_gemini_api_key_here")
    assert not agent.is_configured
    result = agent.analyze_text("Test Hinglish text")
    assert not result["is_available"]
    assert result["status"] == "UNAVAILABLE"
    assert result["detected_category"] is None


def test_agent_empty_text_handling(mock_gemini_client):
    """Verify empty or whitespace-only text returns default Neutral without calling API."""
    agent = ModerationAgent(api_key="valid-test-key", client=mock_gemini_client)
    result = agent.analyze_text("   ")
    assert result["is_available"] is True
    assert result["status"] == "SUCCESS"
    assert result["detected_category"] == "Neutral"
    assert not result["is_identity_targeted"]
    assert not result["requires_human_review"]
    # Verify no API call was made
    mock_gemini_client.models.generate_content.assert_not_called()


def test_agent_successful_structured_response(mock_gemini_client):
    """Verify successful parsing of structured Pydantic response from Gemini SDK."""
    agent = ModerationAgent(api_key="valid-test-key", client=mock_gemini_client)

    # Mock response object with parsed attribute matching GeminiModerationResponse
    expected_response = GeminiModerationResponse(
        detected_category="Offensive",
        is_identity_targeted=False,
        target_group=None,
        severity="Medium",
        explanation="General vulgar slang directed at an individual without identity targeting.",
        contextual_signals=["vulgar slang", "informal insult"],
        requires_human_review=False,
    )
    mock_api_resp = MagicMock()
    mock_api_resp.parsed = expected_response
    mock_api_resp.text = expected_response.model_dump_json()
    mock_gemini_client.models.generate_content.return_value = mock_api_resp

    result = agent.analyze_text("Tu bilkul stupid chutiya hai")

    assert result["is_available"] is True
    assert result["status"] == "SUCCESS"
    assert result["detected_category"] == "Offensive"
    assert result["is_identity_targeted"] is False
    assert result["severity"] == "Medium"
    assert result["requires_human_review"] is False
    assert "vulgar slang" in result["contextual_signals"]


def test_agent_successful_json_text_parsing(mock_gemini_client):
    """Verify fallback parsing when api_response.parsed is None but text is valid JSON."""
    agent = ModerationAgent(api_key="valid-test-key", client=mock_gemini_client)

    json_str = (
        '{"detected_category": "Hate", "is_identity_targeted": true, '
        '"target_group": "Religious Community", "severity": "High", '
        '"explanation": "Communal slur with dehumanizing rhetoric.", '
        '"contextual_signals": ["communal slur"], "requires_human_review": true}'
    )
    mock_api_resp = MagicMock()
    mock_api_resp.parsed = None
    mock_api_resp.text = f"```json\n{json_str}\n```"
    mock_gemini_client.models.generate_content.return_value = mock_api_resp

    result = agent.analyze_text("Targeted communal hate text")

    assert result["is_available"] is True
    assert result["detected_category"] == "Hate"
    assert result["is_identity_targeted"] is True
    assert result["target_group"] == "Religious Community"
    assert result["severity"] == "High"
    assert result["requires_human_review"] is True


def test_agent_api_exception_handling(mock_gemini_client):
    """Verify graceful handling when Gemini API raises an exception (never crashes)."""
    agent = ModerationAgent(api_key="valid-test-key", client=mock_gemini_client)
    mock_gemini_client.models.generate_content.side_effect = RuntimeError("Quota exceeded or network timeout")

    result = agent.analyze_text("Some text")

    assert result["is_available"] is False
    assert result["status"] == "ERROR"
    assert result["detected_category"] is None
    assert result["requires_human_review"] is True
    assert "Quota exceeded" in result["explanation"]


def test_agent_non_fabrication_contract():
    """Verify the agent contract: when unavailable or failed, never fabricate a classification."""
    agent = ModerationAgent(api_key=None)
    result = agent.analyze_text("Suspicious text")
    assert result["detected_category"] is None
    assert result["is_identity_targeted"] is False
    assert result["target_group"] is None
