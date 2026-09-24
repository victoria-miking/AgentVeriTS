# Official OpenAI Responses integration

`OpenAIReasoningClient` uses the official Python SDK, an explicit `https://api.openai.com/v1` base URL, and `OPENAI_API_KEY`. It does not read a relay base URL. Network proxies configured by the operating environment remain an SDK networking concern, separate from the API endpoint.

The repository already contained an initial `previous_response_id` implementation before this migration. This change completes and validates that path rather than claiming the public repository had only manual replay.

## Continuity

The first candidate-assessment call stores a response. The next verification call points to it; every subsequent call points to the immediately preceding completed response. The parent changes after observations and after finalizing each target. Only the new request's text and images are transmitted as `input`. Instructions and the current structured-output schema are provided every time. `store=False` is rejected because this implementation depends on server-side stored responses.

For the paper's GPT-5.6 family, `reasoning.context=all_turns` requests reuse of available compatible reasoning state. This does not guarantee a particular reasoning trajectory or expose hidden reasoning. The flag is omitted for other model families; changing families does not imply reasoning-state compatibility.

The paper's tools are local, read-only functions selected through structured JSON actions. Their observations are new user input items in the same Responses chain. There are no fabricated `function_call_output` items or call IDs: native function-calling is not used by this protocol. Every object in the strict output schemas has `additionalProperties=false` and all properties marked required, with null for unused optional values.

## Failure behavior

The SDK uses a 120-second timeout and two retries by default, both configurable. Transient retries preserve the request and parent ID. An incomplete response, refusal, invalid JSON, missing response ID, context overflow, or missing/expired parent fails explicitly. The client does not erase history, retry without a parent, or switch to Chat Completions. `api_calls.json` records status and available IDs even on failure; existing assessment and completed evidence records are also retained. This is an audit trail, not an automatic crash-resume implementation.

Stored response IDs have service-side retention and account constraints. Runs using unavailable stored state must be restarted explicitly. Earlier context still contributes to model input usage; sending only incremental input does not mean earlier context is free. Data is stored by OpenAI according to the account's API data controls, so configurations that prohibit response storage require a different integration.

## Official references

- [Conversation state](https://developers.openai.com/api/docs/guides/conversation-state)
- [Reasoning models and persisted reasoning](https://developers.openai.com/api/docs/guides/reasoning)
- [Structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs)

## Validation boundary

SDK tests use `httpx.MockTransport` to inspect actual serialized requests and simulate retries, refusal, incomplete output and invalid parent errors. They incur no model charges. An authenticated live API call and full benchmark rerun were not performed for this migration; an official key, model access, pretrained weights and benchmark data are needed for those checks.
