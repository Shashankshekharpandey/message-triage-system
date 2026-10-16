from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

IntentName = Literal[
    "order_tracking", "order_cancellation", "refund_request", "new_order",
    "wrong_item_or_damaged_product", "subscription_management", "product_information",
    "medical_or_health_question", "appointment_booking", "appointment_rescheduling",
    "appointment_cancellation", "service_pricing", "business_hours_or_availability",
    "customer_feedback", "wholesale_or_business_enquiry", "travel_booking",
    "travel_change_or_cancellation", "urgent_travel_disruption", "payment_or_invoice",
    "job_application", "unsubscribe", "ambiguous_or_missing_context", "other",
]
Priority = Literal["low", "normal", "high", "urgent"]
Team = Literal[
    "order_support", "billing_and_refunds", "subscription_support", "product_support",
    "qualified_health_review", "salon_appointments", "salon_services", "travel_support",
    "urgent_travel_support", "finance_review", "wholesale_sales", "recruitment",
    "marketing_or_unsubscribe", "general_support", "human_review",
]
ActionType = Literal[
    "check_tracking", "investigate_delivery", "review_refund", "review_financial_request",
    "manage_subscription_request", "answer_product_question", "qualified_health_review",
    "book_appointment_request", "clarify_appointment_request", "provide_service_information",
    "review_travel_request", "urgent_travel_escalation", "unsubscribe_request",
    "review_job_application", "clarify_context", "review_message", "acknowledge_feedback",
    "request_missing_information", "no_action",
]


class IncomingMessage(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        json_schema_extra={"examples": [{
            "message_id": "demo-1", "brand": "Vitalis Wellness", "channel": "WhatsApp",
            "message": "My order VW-48812 tracking has not changed for four days."
        }]},
    )
    message_id: str | None = Field(default=None, validation_alias=AliasChoices("message_id", "id"))
    brand: str | None = None
    channel: str | None = None
    message: Any = None
    text: Any = None
    timestamp: str | None = Field(default=None, validation_alias=AliasChoices("timestamp", "received_at"))
    conversation_id: str | None = None
    customer_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("message_id", "brand", "channel", "timestamp", "conversation_id", "customer_id", mode="before")
    @classmethod
    def stringify_identifiers(cls, value: Any) -> Any:
        return str(value) if value is not None and not isinstance(value, str) else value


class Intent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: IntentName
    confidence: float = Field(ge=0, le=1)


class ExtractedEntities(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_ids: list[str] = Field(default_factory=list)
    booking_references: list[str] = Field(default_factory=list)
    products_or_services: list[str] = Field(default_factory=list)
    requested_actions: list[str] = Field(default_factory=list)
    dates_times: list[dict[str, str]] = Field(default_factory=list)
    prices: list[str] = Field(default_factory=list)
    contact_details: list[str] = Field(default_factory=list)
    destinations: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    other_details: list[str] = Field(default_factory=list)
    lookup_required: list[str] = Field(default_factory=list)
    attachment_available: bool | None = None


class NextAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_type: ActionType
    description: str
    references: list[str] = Field(default_factory=list)
    automated_workflow_safe: bool = False
    prerequisites: list[str] = Field(default_factory=list)


class TriageResult(BaseModel):
    message_id: str | None
    status: Literal["processed", "failed"]
    brand: str | None
    channel: str | None
    intents: list[Intent]
    entities: ExtractedEntities
    priority: Priority
    assigned_team: Team
    next_actions: list[NextAction]
    confidence: float = Field(ge=0, le=1)
    confidence_basis: str
    extraction_uncertainties: list[str] = Field(default_factory=list)
    human_review_required: bool
    review_reasons: list[str]
    processing_mode: Literal["mock", "llm", "error"]
    processing_time_ms: float | None = None
    token_usage: dict[str, int] | None = None
    error: str | None = None


class BatchRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"messages": [
        {"message_id": "demo-1", "brand": "Vitalis Wellness", "channel": "WhatsApp", "message": "My order VW-48812 is delayed."},
        {"id": "demo-2", "brand": "Hair Studio", "channel": "Instagram", "text": "Are you open on Sunday?"}
    ]}]})
    messages: list[Any]


class BatchResponse(BaseModel):
    received: int
    processed: int
    requiring_review: int
    failed: int
    results: list[TriageResult]
