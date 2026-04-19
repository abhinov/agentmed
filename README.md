# multimodal-bench

An enterprise-grade, general-purpose vision evaluation package designed for robust and flexible multimodal LLM testing.

## Key Features

- **Domain-Agnostic Config:** Provides a generic schema for predictions, confidence scores, and reasoning, dynamically adaptable via `eval_config.json`.
- **Memory-Safe Local Processing:** Handles image loading and base64 conversions safely without memory bloat.
- **Fault-Tolerant Checkpointing:** Supports incremental progress saving, allowing evaluations to seamlessly recover from API rate limits or network issues.

## Getting Started

Check out the `Makefile` commands to install the project and run your first benchmark!
