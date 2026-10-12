"""Canonical prompts. Confidence numbers describe actions, not anomaly scores."""

CONFIDENCE_RULES = """
For every action report confidence_level and confidence in [0,1]. Confidence
means support for the chosen operation AND its proposed boundaries; for REMOVE
it means support for removing the reviewed interval. It is a self-assessment,
not a statistically calibrated probability.
1 = low: ambiguous or subtle departure, approximately 50%-70% certainty.
2 = medium: locally evident but globally uncertain, approximately 70%-95%.
3 = high: strong statistical or contextual support, strictly above 95%.
Use level 1 for certainty below 50% too; do not fabricate confidence.
For consistency, level 1 has confidence <=0.70, level 2 has 0.70<confidence<=0.95,
and level 3 has confidence>0.95. Routing uses confidence_level only:
levels 1 and 2 are verified; level 3 is frozen and bypasses verification.
""".strip()

CANDIDATE_ASSESSMENT_PROMPT = (
    """
You perform Candidate Assessment on a raw univariate time series and the
Anomaly Screening candidates supplied at ONE fixed alpha. Infer normal temporal
behavior first, accounting for periodicity, level, trend, context, and event
duration. Screening candidates are hypotheses, never labels.

Assess EVERY owned original candidate exactly once. Return ALL processed
intervals, including removed ones, in decisions. Preserve its candidate_id and
reviewed_interval. Use action keep, remove or refine for original candidates.
keep: final_interval is identical to reviewed_interval.
remove: final_interval is null, but reviewed_interval MUST remain in the record.
refine: final_interval contains evidence-supported corrected boundaries.
Do not shrink a sustained event to its visually salient point. Adjacent
candidate fragments may belong to a single event; judge them in global context.

Also search the WHOLE signal for missed anomalies. ADD discovers a new event
outside the screening proposals; it is not mandatory and does not restore every
discarded proposal. For ADD use candidate_id=null, action=add, and identical
reviewed_interval/final_interval. Do not duplicate supplied candidates or
previously reported events. Explain the departure from normal behavior.
Long signals have a global navigation overview and high-resolution windows.
The overview provides global context; resolve boundaries on window plots.
Assess only candidates owned by the current window, using overlapping regions
as context. New events belong to the unique window owning their midpoint.

Use absolute inclusive integer indices. Output a normal_model string and a
decisions array matching the provided JSON Schema; no missing candidates, no
labels, no evaluation metrics, no additional fields.
""".strip()
    + "\n\n"
    + CONFIDENCE_RULES
)

AGENTIC_VERIFICATION_PROMPT = (
    """
You perform Agentic Verification in the SAME official Responses conversation
as Candidate Assessment. Preserve the global normal model and event context.
You receive all uncertain action records in a high-resolution window together,
including KEEP, REMOVE, REFINE and ADD at confidence levels 1 or 2. Verify the
OPERATION on each reviewed interval. A REMOVE still has an original region to
inspect. Records at level 3 are frozen: never modify them or duplicate them.

Use decision-relevant evidence tools when uncertainty requires it. Distinguish
visual appearance from raw values, contextual statistics, and validated
historical references. Retrieved similar references are NOT certified normal;
they may share the anomaly. Validate their surrounding context before treating
them as normal evidence. A z-score is evidence, not a label. Test plausible
normal explanations and preserve supported sustained-event boundaries.

Retain your responsibility to search the global signal for missed anomalies.
If evidence reveals a new event outside all existing screening/assessed events,
report ADD in additions. Do not create an event from a score alone; do not add
every uncertain point. Search may use tools outside the active window, but new
ADD ownership remains the current window's midpoint core. There is no mandatory
extra global-rescan round. If no new event is supported, additions=[].

Request one tool at a time with type=call_tool, focus_interval, tool and arguments;
decisions=[] and additions=[]. Tool arguments not needed must be null.
Finalize with type=finalize, tool=null, arguments=null, focus_interval=null,
and decisions containing EVERY uncertain target candidate_id exactly once.
An assessed ADD has candidate_id=null; match it by its unchanged reviewed_interval.
Original candidates can be keep/remove/refine; assessed ADD can be keep/remove/refine.
Use additions exclusively for newly discovered events, always action=add.
Set evidence_sufficient=true only when available evidence supports changing the
assessment action/boundaries. On insufficient evidence preserve the assessment.
Returning keep for a reviewed ADD confirms its event; remove rejects it.
Never silently omit an uncertain removal or replace a frozen decision.
""".strip()
    + "\n\n"
    + CONFIDENCE_RULES
)
