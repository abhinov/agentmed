# multimodal-bench

An enterprise-grade, general-purpose vision evaluation package designed for robust and flexible multimodal LLM testing.

## Key Features

- **Domain-Agnostic Config:** Provides a generic schema for predictions, confidence scores, and reasoning, dynamically adaptable via `eval_config.json`.
- **Memory-Safe Local Processing:** Handles image loading and base64 conversions safely without memory bloat.
- **Fault-Tolerant Checkpointing:** Supports incremental progress saving, allowing evaluations to seamlessly recover from API rate limits or network issues.

## Dataset Preparation

Before evaluating custom datasets with the `make grade` command, you need to map your raw dataset labels to our required format. A generic label formatter script is provided to automate this:

1. Open `scripts/format_labels.py` and update the `CONFIGURATION` block with your raw CSV path and column names.
2. Run the script:
   ```bash
   python scripts/format_labels.py
   ```
This will output a standardized `data/ground_truth/ground_truth.csv` file that the engine requires.

## Getting Started

Check out the `Makefile` commands to install the project and run your first benchmark!
