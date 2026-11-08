"""
Database module for HinglishGuard AI.

Implements SQLAlchemy ORM models and operations for storing moderation audit
records, tracking model & agent predictions, managing the human-in-the-loop review
queue, and recording human moderator determinations.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, Generator, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    desc,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Session,
    relationship,
    sessionmaker,
)

import config

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base declarative class for SQLAlchemy models."""
    pass


class ModerationRecord(Base):
    """SQLAlchemy model for storing every processed moderation request."""

    __tablename__ = "moderation_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    original_text = Column(Text, nullable=False)
    cleaned_text = Column(Text, nullable=True)

    # Supervised Classifier Predictions
    predicted_label = Column(String(50), nullable=False)  # 'Neutral', 'Offensive', 'Hate'
    confidence = Column(Float, nullable=False)
    class_probabilities = Column(Text, nullable=False)  # JSON string: {"Neutral": 0.8, ...}

    # Gemini LLM Moderation Agent Results (optional / nullable)
    gemini_label = Column(String(50), nullable=True)  # 'Neutral', 'Offensive', 'Hate', or None
    gemini_explanation = Column(Text, nullable=True)
    has_disagreement = Column(Boolean, default=False, nullable=False)

    # Policy Engine Outcomes
    policy_action = Column(String(50), nullable=False, default=config.POLICY_ACTION_APPROVE)
    review_status = Column(String(50), nullable=False, default=config.REVIEW_STATUS_RESOLVED)

    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationship to human reviews
    reviews = relationship("HumanReview", back_populates="moderation_record", cascade="all")

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a dictionary representation."""
        probs = {}
        if self.class_probabilities:
            try:
                probs = json.loads(self.class_probabilities)
            except Exception:
                probs = {}

        return {
            "id": self.id,
            "original_text": self.original_text,
            "cleaned_text": self.cleaned_text,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "class_probabilities": probs,
            "gemini_label": self.gemini_label,
            "gemini_explanation": self.gemini_explanation,
            "has_disagreement": self.has_disagreement,
            "policy_action": self.policy_action,
            "review_status": self.review_status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class HumanReview(Base):
    """SQLAlchemy model for recording determinations by human moderators."""

    __tablename__ = "human_reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    moderation_record_id = Column(Integer, ForeignKey("moderation_records.id"), nullable=False, index=True)
    reviewer_decision = Column(String(50), nullable=False)  # 'Neutral', 'Offensive', 'Hate', 'Dismissed'
    reviewer_notes = Column(Text, nullable=True)
    reviewed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationship to parent moderation record
    moderation_record = relationship("ModerationRecord", back_populates="reviews")

    def to_dict(self) -> Dict[str, Any]:
        """Convert human review to a dictionary representation."""
        return {
            "id": self.id,
            "moderation_record_id": self.moderation_record_id,
            "reviewer_decision": self.reviewer_decision,
            "reviewer_notes": self.reviewer_notes,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
        }


# SQLite Engine and Session Factory
engine = create_engine(
    config.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in config.DATABASE_URL else {},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Context manager for safely acquiring and releasing database sessions."""
    db = SessionLocal()
    try:
        yield db
    except Exception as exc:
        db.rollback()
        logger.error(f"Database session error: {exc}", exc_info=True)
        raise
    finally:
        db.close()


def init_db() -> None:
    """Initialize database schema, creating tables if they do not exist."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully.")
    except Exception as exc:
        logger.error(f"Failed to initialize database: {exc}", exc_info=True)
        raise


def save_moderation_record(
    original_text: str,
    predicted_label: str,
    confidence: float,
    class_probabilities: Dict[str, float],
    cleaned_text: Optional[str] = None,
    gemini_label: Optional[str] = None,
    gemini_explanation: Optional[str] = None,
    has_disagreement: bool = False,
    policy_action: str = config.POLICY_ACTION_APPROVE,
    review_status: str = config.REVIEW_STATUS_RESOLVED,
) -> ModerationRecord:
    """Save a moderation record to the database.

    Args:
        original_text: Raw incoming text.
        predicted_label: Model prediction ('Neutral', 'Offensive', 'Hate').
        confidence: Prediction confidence score (0.0 to 1.0).
        class_probabilities: Dict mapping class names to probabilities.
        cleaned_text: Preprocessed text.
        gemini_label: Agent prediction if available.
        gemini_explanation: Agent reasoning if available.
        has_disagreement: Whether classifier and agent differed.
        policy_action: 'APPROVE', 'FLAG', or 'ESCALATE_TO_HUMAN_REVIEW'.
        review_status: 'PENDING' or 'RESOLVED'.

    Returns:
        The saved ModerationRecord instance.
    """
    probs_json = json.dumps(class_probabilities)

    with get_db() as db:
        record = ModerationRecord(
            original_text=original_text,
            cleaned_text=cleaned_text,
            predicted_label=predicted_label,
            confidence=float(confidence),
            class_probabilities=probs_json,
            gemini_label=gemini_label,
            gemini_explanation=gemini_explanation,
            has_disagreement=has_disagreement,
            policy_action=policy_action,
            review_status=review_status,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        # Detach attributes before session closes
        db.expunge(record)
        return record


def get_moderation_history(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """Retrieve historical moderation records ordered by creation date descending.

    Args:
        limit: Maximum records to return.
        offset: Offset for pagination.

    Returns:
        List of moderation record dicts.
    """
    with get_db() as db:
        records = (
            db.query(ModerationRecord)
            .order_by(desc(ModerationRecord.created_at))
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [r.to_dict() for r in records]


def get_pending_review_items(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve moderation records awaiting human review.

    Args:
        limit: Maximum pending records to retrieve.

    Returns:
        List of pending moderation record dicts.
    """
    with get_db() as db:
        records = (
            db.query(ModerationRecord)
            .filter(ModerationRecord.review_status == config.REVIEW_STATUS_PENDING)
            .order_by(desc(ModerationRecord.created_at))
            .limit(limit)
            .all()
        )
        return [r.to_dict() for r in records]


def save_human_review(
    moderation_record_id: int,
    reviewer_decision: str,
    reviewer_notes: Optional[str] = None,
) -> HumanReview:
    """Save a human reviewer's final determination and update record status.

    Does not modify or delete the original record; marks review_status as 'RESOLVED'.

    Args:
        moderation_record_id: Primary key of the parent moderation record.
        reviewer_decision: Final label assigned ('Neutral', 'Offensive', 'Hate', etc.).
        reviewer_notes: Optional notes by the human moderator.

    Returns:
        The saved HumanReview instance.
    """
    with get_db() as db:
        record = db.query(ModerationRecord).filter(ModerationRecord.id == moderation_record_id).first()
        if not record:
            raise ValueError(f"Moderation record with ID {moderation_record_id} does not exist.")

        # Update record review status
        record.review_status = config.REVIEW_STATUS_RESOLVED

        # Create review log entry
        review = HumanReview(
            moderation_record_id=moderation_record_id,
            reviewer_decision=reviewer_decision,
            reviewer_notes=reviewer_notes,
        )
        db.add(review)
        db.commit()
        db.refresh(review)
        db.expunge(review)
        return review


def get_database_summary() -> Dict[str, Any]:
    """Retrieve aggregate statistics from the moderation records.

    Returns:
        Dict with keys: total_records, pending_reviews, resolved_reviews,
        disagreements_count, label_counts, and policy_action_counts.
    """
    with get_db() as db:
        records = db.query(ModerationRecord).all()
        total = len(records)
        pending = sum(1 for r in records if r.review_status == config.REVIEW_STATUS_PENDING)
        resolved = sum(1 for r in records if r.review_status == config.REVIEW_STATUS_RESOLVED)
        disagreements = sum(1 for r in records if r.has_disagreement)

        label_counts = {"Neutral": 0, "Offensive": 0, "Hate": 0}
        action_counts = {}
        for r in records:
            if r.predicted_label in label_counts:
                label_counts[r.predicted_label] += 1
            else:
                label_counts[r.predicted_label] = 1

            act = r.policy_action or "UNKNOWN"
            action_counts[act] = action_counts.get(act, 0) + 1

        return {
            "total_records": total,
            "pending_reviews": pending,
            "resolved_reviews": resolved,
            "disagreements_count": disagreements,
            "label_counts": label_counts,
            "policy_action_counts": action_counts,
        }


def update_moderation_record_with_agent(
    record_id: int,
    gemini_label: Optional[str],
    gemini_explanation: Optional[str],
    has_disagreement: bool,
    policy_action: str,
    review_status: str,
) -> Optional[ModerationRecord]:
    """Update an existing moderation record with Gemini agent analysis and revised policy action.

    Args:
        record_id: Primary key ID of the existing record.
        gemini_label: LLM detected category ('Neutral', 'Offensive', 'Hate', or None).
        gemini_explanation: Contextual explanation from Gemini.
        has_disagreement: True if classifier and agent differed.
        policy_action: Revised policy action (APPROVE, FLAG, ESCALATE_TO_HUMAN_REVIEW).
        review_status: Revised status (PENDING, RESOLVED, etc.).

    Returns:
        Updated ModerationRecord instance or None if not found.
    """
    with get_db() as db:
        record = db.query(ModerationRecord).filter(ModerationRecord.id == record_id).first()
        if not record:
            return None
        record.gemini_label = gemini_label
        record.gemini_explanation = gemini_explanation
        record.has_disagreement = has_disagreement
        record.policy_action = policy_action
        record.review_status = review_status
        db.commit()
        db.refresh(record)
        db.expunge(record)
        return record


def get_records_by_status(status_filter: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieve moderation records filtered by review_status.

    Args:
        status_filter: 'PENDING', 'RESOLVED', or None (for all records).
        limit: Maximum number of records to retrieve.

    Returns:
        List of moderation record dicts with any linked human reviews included.
    """
    with get_db() as db:
        query = db.query(ModerationRecord)
        if status_filter and status_filter.upper() != "ALL":
            query = query.filter(ModerationRecord.review_status == status_filter.upper())
        records = query.order_by(desc(ModerationRecord.created_at)).limit(limit).all()

        results = []
        for r in records:
            r_dict = r.to_dict()
            r_dict["human_reviews"] = [rev.to_dict() for rev in r.reviews]
            results.append(r_dict)
        return results


def get_human_review_statistics() -> Dict[str, Any]:
    """Retrieve summary statistics of human moderation decisions.

    Returns:
        Dict with keys: pending_count, resolved_count, total_reviews,
        and human_decision_distribution (counts per final decision).
    """
    with get_db() as db:
        records = db.query(ModerationRecord).all()
        pending = sum(1 for r in records if r.review_status == config.REVIEW_STATUS_PENDING)
        resolved = sum(1 for r in records if r.review_status == config.REVIEW_STATUS_RESOLVED)

        reviews = db.query(HumanReview).all()
        total_rev = len(reviews)
        decision_dist: Dict[str, int] = {}
        for rev in reviews:
            dec = rev.reviewer_decision
            decision_dist[dec] = decision_dist.get(dec, 0) + 1

        return {
            "pending_count": pending,
            "resolved_count": resolved,
            "total_reviews": total_rev,
            "human_decision_distribution": decision_dist,
        }


def get_all_moderation_analytics_data() -> Dict[str, Any]:
    """Retrieve full analytics dataset directly from SQLite database.

    Returns:
        Dict with keys:
            - records: list of all ModerationRecord dicts with attached human reviews.
            - total_count: int
            - label_counts: dict (Neutral, Offensive, Hate)
            - action_counts: dict (APPROVE, FLAG, ESCALATE)
            - pending_count: int
            - resolved_count: int
            - disagreements_count: int
            - avg_confidence: float (overall average)
            - avg_confidence_by_label: dict of label -> float
            - human_decisions: dict of reviewer decisions
            - daily_activity: dict of YYYY-MM-DD -> count
    """
    with get_db() as db:
        records = db.query(ModerationRecord).order_by(desc(ModerationRecord.created_at)).all()
        reviews = db.query(HumanReview).all()

        total = len(records)
        if total == 0:
            return {
                "records": [],
                "total_count": 0,
                "label_counts": {"Neutral": 0, "Offensive": 0, "Hate": 0},
                "action_counts": {},
                "pending_count": 0,
                "resolved_count": 0,
                "disagreements_count": 0,
                "avg_confidence": 0.0,
                "avg_confidence_by_label": {"Neutral": 0.0, "Offensive": 0.0, "Hate": 0.0},
                "human_decisions": {},
                "daily_activity": {},
            }

        label_counts = {"Neutral": 0, "Offensive": 0, "Hate": 0}
        conf_sums = {"Neutral": 0.0, "Offensive": 0.0, "Hate": 0.0}
        action_counts: Dict[str, int] = {}
        daily_activity: Dict[str, int] = {}
        pending_count = 0
        resolved_count = 0
        disagreements_count = 0
        total_conf = 0.0

        record_dicts = []
        for r in records:
            r_dict = r.to_dict()
            r_dict["human_reviews"] = [rev.to_dict() for rev in r.reviews]
            record_dicts.append(r_dict)

            lbl = r.predicted_label or "Neutral"
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
            conf = float(r.confidence or 0.0)
            total_conf += conf
            conf_sums[lbl] = conf_sums.get(lbl, 0.0) + conf

            act = r.policy_action or "UNKNOWN"
            action_counts[act] = action_counts.get(act, 0) + 1

            if r.review_status == config.REVIEW_STATUS_PENDING:
                pending_count += 1
            elif r.review_status == config.REVIEW_STATUS_RESOLVED:
                resolved_count += 1

            if r.has_disagreement:
                disagreements_count += 1

            if r.created_at:
                day_str = r.created_at.strftime("%Y-%m-%d")
                daily_activity[day_str] = daily_activity.get(day_str, 0) + 1

        avg_conf = total_conf / total if total > 0 else 0.0
        avg_by_label = {}
        for l_name, count in label_counts.items():
            avg_by_label[l_name] = (conf_sums.get(l_name, 0.0) / count) if count > 0 else 0.0

        human_decisions: Dict[str, int] = {}
        for rev in reviews:
            dec = rev.reviewer_decision
            human_decisions[dec] = human_decisions.get(dec, 0) + 1

        return {
            "records": record_dicts,
            "total_count": total,
            "label_counts": label_counts,
            "action_counts": action_counts,
            "pending_count": pending_count,
            "resolved_count": resolved_count,
            "disagreements_count": disagreements_count,
            "avg_confidence": avg_conf,
            "avg_confidence_by_label": avg_by_label,
            "human_decisions": human_decisions,
            "daily_activity": daily_activity,
        }


def update_review_status(moderation_record_id: int, new_status: str) -> bool:
    """Update review status of a moderation record (e.g., 'PENDING', 'RESOLVED', 'SKIPPED').

    Args:
        moderation_record_id: Primary key of the record.
        new_status: New status string.

    Returns:
        True if updated successfully, False if record not found.
    """
    with get_db() as db:
        record = db.query(ModerationRecord).filter(ModerationRecord.id == moderation_record_id).first()
        if not record:
            return False
        record.review_status = new_status
        db.commit()
        return True
