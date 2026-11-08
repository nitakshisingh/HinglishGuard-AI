"""
Gemini LLM Moderation Agent Module for HinglishGuard AI.

Leverages the Google GenAI SDK (`google.genai`) to provide deep contextual reasoning,
identity target detection, sarcasm/negation analysis, and nuanced code-mixed
linguistic interpretation. Returns structured moderation assessments with safe fallbacks
when the API key is missing or calls fail. Never fabricates analysis.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

import config

logger = logging.getLogger(__name__)

# Structured output schema using Pydantic
class GeminiModerationResponse(BaseModel):
    """Pydantic schema enforcing structured JSON output from Gemini."""

    detected_category: str = Field(
        ...,
        description="Must be one of: 'Neutral', 'Offensive', 'Hate', 'Uncertain'.",
    )
    is_identity_targeted: bool = Field(
        ...,
        description="True if the text targets individuals/groups based on religion, caste, gender, ethnicity, etc.",
    )
    target_group: Optional[str] = Field(
        None,
        description="Targeted identity group name if applicable (e.g. 'Religious Community', 'Caste', 'Gender'), else null.",
    )
    severity: str = Field(
        ...,
        description="Must be one of: 'Low', 'Medium', 'High'.",
    )
    explanation: str = Field(
        ...,
        description="Concise, objective explanation of the semantic context, Hinglish slang, and intent.",
    )
    contextual_signals: List[str] = Field(
        default_factory=list,
        description="List of detected linguistic signals (e.g., ['communal slur', 'negated insult', 'sarcasm']).",
    )
    requires_human_review: bool = Field(
        ...,
        description="True if the context is ambiguous, borderline, highly sensitive, or requires human judgment.",
    )


AGENT_SYSTEM_PROMPT = """You are an expert bilingual content moderation agent specialized in code-mixed Hindi-English (Hinglish) social media text.

Your role is to perform contextual linguistic analysis to distinguish between:
1. Neutral: Benign conversation, friendly banter, civil political debate, constructive disagreements, or neutral reporting.
2. Offensive: General vulgarity, profanity, crude insults, obscene slang, or coarse venting WITHOUT identity/group targeting.
3. Hate: Targeted attacks, dehumanization, threats of violence, incitement, or discrimination directed against an individual or group based on protected characteristics (religion, caste, ethnicity, nationality, gender, sexual orientation).
4. Uncertain: Ambiguous statements where intent cannot be reliably deduced without external thread context.

CRITICAL MODERATION RULES:
- Do NOT treat every profanity or vulgar word as hate speech. General swearing (e.g., 'kutta', 'chutiya', 'stupid', 'bhadwa') directed at personal actions is Offensive, NOT Hate.
- Mark 'is_identity_targeted = True' and 'detected_category = Hate' ONLY when protected social identities, religious groups, or communities are attacked, dehumanized, or threatened.
- Account for negation (e.g., 'hate nahi karta' means not hating).
- Identify code-mixed euphemisms, coded slurs, and sarcastic phrases.
- Provide objective, factual explanations.
"""


class ModerationAgent:
    """Agent orchestrating LLM-based contextual content analysis via Google Gemini."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, client: Any = None):
        """Initialize the moderation agent.

        Args:
            api_key: Optional explicit API key; if None, defaults to config.GEMINI_API_KEY.
                     Pass explicit "" or "your_gemini_api_key_here" to simulate an unconfigured agent.
            model_name: Optional model name; defaults to config.GEMINI_MODEL.
            client: Optional pre-configured GenAI Client (useful for dependency injection / mocking).
        """
        self.api_key = config.GEMINI_API_KEY if api_key is None else api_key
        self.model_name = model_name or config.GEMINI_MODEL
        self.client = client

        # Initialize client if not provided and valid key exists
        if self.client is None and self.is_configured:
            try:
                from google import genai
                # Pass api_version='v1beta' in http_options so system_instruction & developer instructions are accepted
                self.client = genai.Client(api_key=self.api_key, http_options={"api_version": "v1beta"})
            except Exception as exc:
                logger.warning(f"Failed to initialize google.genai Client: {exc}")
                self.client = None

    @property
    def is_configured(self) -> bool:
        """Check whether the agent has a valid non-placeholder API key configured."""
        return bool(
            self.api_key
            and self.api_key.strip()
            and self.api_key != "your_gemini_api_key_here"
        )

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Perform contextual semantic analysis of Hinglish text using Gemini.

        If the API key is not configured or an error occurs, returns a structured
        fallback with 'is_available = False' without fabricating predictions.

        Args:
            text: Raw input text.

        Returns:
            Dictionary containing agent analysis and status metadata.
        """
        if not text or not text.strip():
            return {
                "is_available": True,
                "detected_category": "Neutral",
                "is_identity_targeted": False,
                "target_group": None,
                "severity": "Low",
                "explanation": "Empty or blank text submitted.",
                "contextual_signals": ["empty_input"],
                "requires_human_review": False,
                "status": "SUCCESS",
            }

        # Safe fallback when API key is missing or unconfigured
        if not self.is_configured:
            return {
                "is_available": False,
                "detected_category": None,
                "is_identity_targeted": False,
                "target_group": None,
                "severity": "Low",
                "explanation": "Gemini API key is not configured. Contextual LLM analysis was skipped.",
                "contextual_signals": ["api_key_missing"],
                "requires_human_review": False,
                "status": "UNAVAILABLE",
            }

        # Execute API call using google.genai SDK
        try:
            from google.genai import types

            user_prompt = f"Analyze the following code-mixed text for content moderation:\n\"\"\"{text}\"\"\""

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=AGENT_SYSTEM_PROMPT,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=GeminiModerationResponse,
                ),
            )

            # Parse structured JSON output
            if hasattr(response, "parsed") and isinstance(response.parsed, GeminiModerationResponse):
                validated = response.parsed
            elif hasattr(response, "parsed") and isinstance(response.parsed, dict):
                validated = GeminiModerationResponse(**response.parsed)
            else:
                raw_text = response.text.strip() if hasattr(response, "text") and response.text else "{}"
                # Strip markdown codeblock backticks if present
                if raw_text.startswith("```"):
                    raw_text = raw_text.strip("`")
                    if raw_text.lower().startswith("json"):
                        raw_text = raw_text[4:].strip()
                parsed_json = json.loads(raw_text)
                validated = GeminiModerationResponse(**parsed_json)

            return {
                "is_available": True,
                "detected_category": validated.detected_category,
                "is_identity_targeted": validated.is_identity_targeted,
                "target_group": validated.target_group,
                "severity": validated.severity,
                "explanation": validated.explanation,
                "contextual_signals": validated.contextual_signals,
                "requires_human_review": validated.requires_human_review,
                "status": "SUCCESS",
            }

        except Exception as exc:
            logger.error(f"Gemini moderation agent call failed: {exc}", exc_info=True)
            return {
                "is_available": False,
                "detected_category": None,
                "is_identity_targeted": False,
                "target_group": None,
                "severity": "Low",
                "explanation": f"LLM moderation agent error: {str(exc)}",
                "contextual_signals": ["agent_error"],
                "requires_human_review": True,
                "status": "ERROR",
            }


# Default singleton instance
_DEFAULT_AGENT = None


def get_default_moderation_agent() -> ModerationAgent:
    """Return singleton instance of ModerationAgent."""
    global _DEFAULT_AGENT
    if _DEFAULT_AGENT is None:
        _DEFAULT_AGENT = ModerationAgent()
    return _DEFAULT_AGENT


def analyze_with_agent(text: str) -> Dict[str, Any]:
    """Reusable utility to analyze text with the default moderation agent.

    Args:
        text: Input string.

    Returns:
        Structured analysis dictionary.
    """
    agent = get_default_moderation_agent()
    return agent.analyze_text(text)
