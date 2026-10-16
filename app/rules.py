import re
from typing import Any

from app.models import ExtractedEntities, IncomingMessage, Intent, NextAction, TriageResult

INJECTION = re.compile(r"\b(ignore|override|bypass)\b.{0,100}\b(instructions?|system|policy|admin|administrator|refund|review)\b|\b(suppress|disable)\b.{0,50}\b(escalation|review|logging)\b", re.I)
ORDER_ID = re.compile(r"\b(?:VW|VOY|HS)-?\d{4,}\b", re.I)
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")
PRICE = re.compile(r"\b(?:\$|₹|£|€|USD\s*|INR\s*)\d[\d,]*(?:\.\d{1,2})?\b", re.I)


def normalized(raw: dict[str, Any]) -> tuple[IncomingMessage, str]:
    msg = IncomingMessage.model_validate(raw)
    value = msg.message if msg.message is not None else msg.text
    text = value if isinstance(value, str) else "" if value is None else str(value)
    return msg, text.strip()


def make_result(raw: dict[str, Any], *, mode: str = "mock", draft: dict | None = None,
                duration: float | None = None, tokens: dict | None = None,
                error: str | None = None, confidence_threshold: float = 0.65) -> TriageResult:
    if raw.get("_parse_error"):
        return TriageResult(message_id=None, status="failed", brand=None, channel=None,
            intents=[Intent(name="ambiguous_or_missing_context", confidence=0)], entities=ExtractedEntities(),
            priority="normal", assigned_team="human_review",
            next_actions=[NextAction(action_type="review_message", description="Correct the malformed input record")],
            confidence=0, confidence_basis="JSON parsing failed; no classification was attempted",
            human_review_required=True, review_reasons=["invalid_input"], processing_mode="error",
            processing_time_ms=duration, error=str(raw["_parse_error"]))
    try:
        msg, text = normalized(raw)
    except Exception as exc:
        return TriageResult(message_id=str(raw.get("message_id")) if raw.get("message_id") is not None else None,
            status="failed", brand=None, channel=None, intents=[Intent(name="ambiguous_or_missing_context", confidence=0)],
            entities=ExtractedEntities(), priority="normal", assigned_team="human_review",
            next_actions=[NextAction(action_type="review_message", description="Review malformed input record")],
            confidence=0, confidence_basis="Input validation failed", extraction_uncertainties=[str(exc)],
            human_review_required=True, review_reasons=["invalid_input"], processing_mode="error", processing_time_ms=duration,
            error="Invalid message record")
    injected = bool(INJECTION.search(text))
    if draft is None:
        draft = deterministic_draft(msg, text)
        mode = "mock"
    try:
        intents = [Intent.model_validate(x) for x in draft.get("intents", [])]
        if not intents:
            intents = [Intent(name="ambiguous_or_missing_context", confidence=0.35)]
        entities = ExtractedEntities.model_validate(draft.get("entities", {}))
        actions = [NextAction.model_validate(x) for x in draft.get("next_actions", [])]
        priority = draft.get("priority", "normal")
        team = draft.get("assigned_team", "human_review")
        confidence = float(draft.get("confidence", min(i.confidence for i in intents)))
        basis = str(draft.get("confidence_basis", "Rule-based evidence match; not statistically calibrated"))
        uncertainties = list(draft.get("extraction_uncertainties", []))
    except Exception:
        return make_result(raw, mode="error", duration=duration, error="Classifier output failed schema validation")
    reasons = list(draft.get("review_reasons", []))
    review = bool(draft.get("human_review_required", False))
    if injected:
        review, team, priority = True, "human_review", "high"
        reasons.append("suspected_prompt_injection")
        actions.insert(0, NextAction(action_type="review_message", description="Review suspected instruction override; do not authorize requested actions"))
    names = {i.name for i in intents}
    if "medical_or_health_question" in names:
        review, team = True, "qualified_health_review"
        reasons.append("medical_question_requires_qualified_review")
        if not any(a.action_type == "qualified_health_review" for a in actions):
            actions.append(NextAction(action_type="qualified_health_review", description="Refer to a qualified health professional; do not provide medical advice"))
    if names & {"refund_request", "payment_or_invoice"}:
        review = True
        if not injected and team not in {"finance_review", "billing_and_refunds"}: team = "finance_review"
        reasons.append("financial_action_requires_authorized_human_review")
        for action in actions:
            action.automated_workflow_safe = False
    if "urgent_travel_disruption" in names:
        review, team, priority = True, "urgent_travel_support", "urgent"
        reasons.append("urgent_travel_disruption_requires_prompt_escalation")
    if "ambiguous_or_missing_context" in names:
        review = True; reasons.append("missing_context_prevents_safe_action")
        for action in actions: action.automated_workflow_safe = False
    if confidence < confidence_threshold:
        review = True; reasons.append("below_configured_confidence_threshold")
    if not text:
        review = True; names = {"ambiguous_or_missing_context"}; team = "human_review"
        intents = [Intent(name="ambiguous_or_missing_context", confidence=0.99)]
        reasons.append("message_text_missing_or_empty")
        actions = [NextAction(action_type="request_missing_information", description="Request the message text before triage")]
    if error:
        return TriageResult(message_id=msg.message_id, status="failed", brand=msg.brand, channel=msg.channel,
            intents=[Intent(name="ambiguous_or_missing_context", confidence=0)], entities=ExtractedEntities(), priority="normal",
            assigned_team="human_review", next_actions=[NextAction(action_type="review_message", description="Retry or inspect failed triage")],
            confidence=0, confidence_basis="Provider or validation failure; no trusted classification produced", extraction_uncertainties=[],
            human_review_required=True, review_reasons=["processing_failure"], processing_mode="error", processing_time_ms=duration, error=error)
    return TriageResult(message_id=msg.message_id, status="processed", brand=msg.brand, channel=msg.channel,
        intents=intents, entities=entities, priority=priority, assigned_team=team, next_actions=actions,
        confidence=confidence, confidence_basis=basis, extraction_uncertainties=uncertainties,
        human_review_required=review, review_reasons=sorted(set(reasons)), processing_mode=mode,
        processing_time_ms=duration, token_usage=tokens)


def deterministic_draft(msg: IncomingMessage, text: str) -> dict:
    t = text.lower()
    intents: list[tuple[str, float]] = []
    def add(name: str, condition: bool, confidence: float = .78):
        if condition and name not in [n for n, _ in intents]: intents.append((name, confidence))
    add("unsubscribe", any(x in t for x in ("unsubscribe", "stop messages", "remove me")))
    add("urgent_travel_disruption", (any(x in t for x in ("stranded", "cancelled connection", "canceled connection", "missed connection", "airport")) and any(x in t for x in ("elderly", "wheelchair", "stranded", "cancelled", "canceled"))) or ("urgent" in t and any(x in t for x in ("airport", "connection", "flight"))), .90)
    add("medical_or_health_question", any(x in t for x in ("medication", "medicine", "blood pressure", "interaction", "capsule safe")))
    add("refund_request", "refund" in t or "money back" in t)
    add("payment_or_invoice", any(x in t for x in ("invoice", "double charge", "charged twice", "payment", "bank details", "how do we pay", "how do i pay")))
    add("wrong_item_or_damaged_product", any(x in t for x in ("damaged", "broken", "wrong item", "wrong strength", "250 mg", "not sealed", "unsealed")))
    add("order_tracking", any(x in t for x in ("tracking", "delivery", "shipment", "stuck", "unchanged")))
    add("order_cancellation", "cancel" in t and "order" in t and "subscription" not in t)
    add("subscription_management", "subscription" in t)
    add("new_order", any(x in t for x in ("place an order", "new order", "also want", "also order", "still like to order", "like to order")))
    add("appointment_rescheduling", any(x in t for x in ("reschedule", "move my appointment", "move my thursday appointment", "move my", "change my appointment", "shift my appointment", "shift it to", "kal kar sakte")))
    add("appointment_cancellation", "cancel" in t and any(x in t for x in ("appointment", "salon", "booking")))
    add("appointment_booking", any(x in t for x in ("book an appointment", "book appointment", "appointment slot", "wants to book")))
    add("service_pricing", (bool(re.search(r"\b(price|pricing|cost|charges|kitna|rate|balayage|keratin)\b", t)) or ("membership" in t and ("salon" in t or "sister" in t))) and not any(x in t for x in ("bulk", "wholesale", "pharmacies")))
    add("business_hours_or_availability", any(x in t for x in ("opening hours", "open on", "are you open", "availability")))
    add("wholesale_or_business_enquiry", any(x in t for x in ("wholesale", "pharmacies", "bulk order")))
    add("job_application", any(x in t for x in ("resume", "cv attached", "job application", "apply for")))
    add("customer_feedback", any(x in t for x in ("great service", "thank you", "loved", "excellent", "wonderful", "amazing job", "obsessed with")))
    add("travel_change_or_cancellation", any(x in t for x in ("change my flight", "change booking", "change the booking", "change my return", "cancel my flight")))
    add("travel_booking", any(x in t for x in ("travel", "flight", "stopover", "fare", "fares", "booking", "toronto to", "parents")) and not bool(re.search(r"\b(salon|hair studio|balayage|keratin)\b", t)))
    add("product_information", (any(x in t for x in ("what is", "ingredients", "tell me about", "product information", "what are")) or ("magnesium" in t and any(x in t for x in ("what", "benefit")) and not any(x in t for x in ("safe", "medication", "interaction")))) and not any(n == "wrong_item_or_damaged_product" for n, _ in intents))
    if any(x in t for x in ("other one", "which one", "that one")) and len(t.split()) < 15: add("ambiguous_or_missing_context", True, .42)
    if re.search(r"\b(work came up|sorry.*work|something came up)\b", t) and not any(n.startswith("appointment_") for n, _ in intents): add("ambiguous_or_missing_context", True, .45)
    if not intents: add("ambiguous_or_missing_context", True, .35)
    names = [n for n, _ in intents]
    if "job_application" in names:
        intents = [(name, confidence) for name, confidence in intents if name != "travel_booking"]
        names = [n for n, _ in intents]
    team = "general_support"; priority = "normal"; action = "review_message"; desc = "Review the message and route it to the appropriate team"
    if "urgent_travel_disruption" in names: team, priority, action, desc = "urgent_travel_support", "urgent", "urgent_travel_escalation", "Escalate promptly to urgent travel support"
    elif "medical_or_health_question" in names: team, action, desc = "qualified_health_review", "qualified_health_review", "Refer the question to a qualified health professional"
    elif "refund_request" in names: team, action, desc = "billing_and_refunds", "review_refund", "Route refund request for authorized review"
    elif "payment_or_invoice" in names: team, action, desc = "finance_review", "review_financial_request", "Review payment or invoice request; do not execute payment"
    elif "wholesale_or_business_enquiry" in names: team, action, desc = "wholesale_sales", "review_message", "Route wholesale enquiry to sales"
    elif "job_application" in names: team, action, desc = "recruitment", "review_job_application", "Route application to recruitment"
    elif "unsubscribe" in names: team, action, desc = "marketing_or_unsubscribe", "unsubscribe_request", "Record unsubscribe request for authorized workflow"
    elif any(n.startswith("appointment_") for n in names): team, action, desc = "salon_appointments", "clarify_appointment_request", "Review appointment request and confirm the requested change"
    elif "service_pricing" in names: team, action, desc = "salon_services", "provide_service_information", "Provide service pricing information"
    elif "business_hours_or_availability" in names: team, action, desc = ("salon_services" if "hair" in (msg.brand or "").lower() else "travel_support"), "provide_service_information", "Check the relevant business hours or availability"
    elif "travel_change_or_cancellation" in names or "travel_booking" in names: team, action, desc = "travel_support", "review_travel_request", "Review travel request and required booking details"
    elif "refund_request" in names: team = "billing_and_refunds"
    elif "wrong_item_or_damaged_product" in names: team, action, desc = "order_support", "investigate_delivery", "Investigate reported product or fulfillment issue"
    elif "order_tracking" in names: team, action, desc = "order_support", "check_tracking", "Check shipment status using the supplied order reference"
    elif "subscription_management" in names: team, action, desc = "subscription_support", "manage_subscription_request", "Review subscription change request"
    elif "product_information" in names: team, action, desc = "product_support", "answer_product_question", "Provide supported product information"
    elif "customer_feedback" in names: team, action, desc = "general_support", "acknowledge_feedback", "Acknowledge and route customer feedback"
    elif "ambiguous_or_missing_context" in names: team, action, desc = "human_review", "clarify_context", "Ask the customer for the missing context"
    order_ids = list(dict.fromkeys(ORDER_ID.findall(text)))
    refs = order_ids + list(dict.fromkeys(re.findall(r"\b[A-Z0-9]{6,10}\b", text)))
    booking_refs = list(dict.fromkeys(re.findall(r"\bVYG[A-Z0-9]{4,8}\b", text, re.I)))
    products = [x for x in ("magnesium", "balayage", "keratin", "500 mg", "250 mg") if x in t]
    destinations = re.findall(r"\b(?:to|in|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", text)
    month_names = {"jan", "january", "feb", "february", "mar", "march", "apr", "april", "may", "jun", "june", "jul", "july", "aug", "august", "sep", "september", "oct", "october", "nov", "november", "dec", "december"}
    destinations = [d for d in destinations if d.lower() not in month_names]
    date_time_entities = [{"text": m.group(0), "month": m.group(1)} for m in re.finditer(r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?\b", text, re.I)]
    date_time_entities += [{"text": x, "month": "unknown"} for x in re.findall(r"\b(?:on\s+)?the\s+(\d{1,2})(?:st|nd|rd|th)?\b", t)]
    date_time_entities += [{"text": m.group(0), "date": "unspecified"} for m in re.finditer(r"\b(?:Saturday|Sunday|Monday|Tuesday|Wednesday|Thursday|Friday)\b.{0,12}\b\d{1,2}(?::\d{2})?\s*(?:am|pm|baje)?\b", text, re.I)]
    durations = re.findall(r"\b\d+\s+(?:hours?|days?)\b", t)
    entities = ExtractedEntities(order_ids=order_ids, booking_references=booking_refs, products_or_services=products,
        requested_actions=[n for n in names], prices=PRICE.findall(text), contact_details=PHONE.findall(text),
        destinations=list(dict.fromkeys(destinations)), other_details=durations,
        lookup_required=["order_system_lookup"] if "order_tracking" in names else [],
        attachment_available=False if "pdf" in t or "attached" in t else None,
        dates_times=date_time_entities)
    reason_match = re.search(r"because\s+([^,.]+)", text, re.I)
    if reason_match: entities.reasons.append(reason_match.group(1).strip())
    score = min(c for _, c in intents)
    if any(x in t for x in ("charged twice", "third time", "resolve it today")): priority = "high"
    if len(intents) > 1: score = min(score, .72)
    if INJECTION.search(text): score = min(score, .55)
    actions = [{"action_type": action, "description": desc, "references": refs,
                "automated_workflow_safe": action in {"check_tracking", "answer_product_question", "provide_service_information", "acknowledge_feedback", "unsubscribe_request"},
                "prerequisites": ["booking reference"] if "travel" in action or "travel_" in action else []}]
    # Preserve actionable work for secondary intents instead of collapsing a multi-request message.
    secondary = {
        "service_pricing": ("provide_service_information", "Provide the requested service pricing or membership information"),
        "wrong_item_or_damaged_product": ("investigate_delivery", "Investigate the reported product or fulfillment issue"),
        "new_order": ("request_missing_information", "Collect the requested product, variant, and quantity for a new order"),
        "appointment_booking": ("book_appointment_request", "Check availability for the requested appointment"),
        "appointment_rescheduling": ("clarify_appointment_request", "Confirm the appointment and requested replacement time"),
        "refund_request": ("review_refund", "Route the refund request for authorized review"),
        "payment_or_invoice": ("review_financial_request", "Review the financial request; do not execute payment"),
        "customer_feedback": ("acknowledge_feedback", "Record and route the customer feedback"),
    }
    for name in names:
        if name in secondary:
            action_name, action_desc = secondary[name]
            if not any(existing["action_type"] == action_name for existing in actions):
                actions.append({"action_type": action_name, "description": action_desc, "references": refs,
                    "automated_workflow_safe": action_name == "acknowledge_feedback", "prerequisites": []})
    return {"intents": [{"name": n, "confidence": c} for n, c in intents], "entities": entities.model_dump(),
        "priority": priority, "assigned_team": team, "next_actions": actions, "confidence": score,
        "confidence_basis": "Deterministic phrase and pattern evidence; heuristic, not statistically calibrated",
        "extraction_uncertainties": ["Request interpretation may need customer context"] if "ambiguous_or_missing_context" in names else [],
        "human_review_required": any(n in names for n in ("refund_request", "payment_or_invoice", "medical_or_health_question", "ambiguous_or_missing_context", "urgent_travel_disruption")),
        "review_reasons": []}
