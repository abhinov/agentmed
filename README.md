# 🏥 AgentMed: Clinical AI Copilot & Multi-Agent Orchestrator

**AgentMed** is a high-fidelity, fault-tolerant evaluation engine and clinical AI Copilot. It is designed to benchmark and orchestrate frontier Vision-Language Models (VLMs) on complex clinical imagery, acting as a multimodal diagnostic auditor. 

While standard benchmarks evaluate text-based clinical chat, AgentMed tackles the multimodal equivalent—evaluating spatial explainability, AI hallucination safety, and Multi-Agent consensus on high-stakes medical scans (IDRiD Dataset).

**Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/beabhinov/agentmed)

---

## 🚀 Business Impact & Engineering Value (Why This Matters)
Standard AI deployments often force a choice between **Cost** (using smaller, cheaper models) and **Accuracy/Safety** (using massive, expensive proprietary models). AgentMed solves this through a dynamic **Multi-Agent Maker-Checker Architecture**.

### 1. Accuracy Uplift via Multi-Agent Consensus
By enforcing a multi-agent debate on ambiguous scans, the AgentMed pipeline achieves a significant accuracy uplift over single-agent baselines.
* **Baseline (Single Agent GPT-4o-mini):** 66.59% Accuracy
* **AgentMed Pipeline (Llama-90B + GPT-4o):** **72.31% Accuracy** (+5.72% absolute uplift)

### 2. Optimizing Unit Economics (Doing More with Less)
Despite using a heavy proprietary model (GPT-4o) for second opinions, the overall pipeline is **cheaper** than running a lightweight proprietary model across the entire dataset. 
* **Baseline Fleet Cost:** $0.29
* **AgentMed Fleet Cost:** **$0.27**

**How?** We utilize a powerful open-source model (**Llama-3.2-90B**) as the frontline "Maker" for the bulk of the workload (handling 62.4% of cases autonomously). The expensive proprietary "Checker" (**GPT-4o-mini**) is only invoked strategically when the Maker exhibits low confidence or detects high-risk pathology.

### 3. Continuous Risk Optimization (Adaptive Thresholds)
Rather than relying on static, hardcoded rules, AgentMed uses an **Exponential Moving Average (EMA)** to track the open-source model's historical confidence. The system dynamically self-adjusts: if the frontline model shows systemic overconfidence, the routing threshold automatically tightens to protect patient safety. If it proves highly reliable, the threshold relaxes to further drive down API costs.

### 4. Clinical Safety & Human-in-the-Loop Triage
AgentMed treats Artificial Intelligence like a **Medical Resident**, while the human remains the **Attending Physician**. 
* **Safe Auto-Triage Rate (62.42%):** Cases where models reached high-confidence consensus.
* **Human Escalation Rate (37.58%):** Cases resulting in a "Hard Deferral," meaning the AI disagreed or lacked confidence, automatically routing the scan to a human doctor.

---

## 🧠 System Architecture

Our architecture proves that a **Multi-Agent "Maker-Checker" loop** drastically reduces hallucination rates in multimodal tasks. 

1. **The Primary AI (Maker):** Meta's open-weights `Llama-3.2-90B-Vision-Instruct` analyzes the raw scan, providing an initial diagnostic grade and a native confidence score.
2. **Dynamic Thresholding Engine:** The system parses the Maker's confidence. To ensure resilience against varying model output formats (e.g., raw floats like `0.85` vs. integers like `85`), the pipeline dynamically updates its benchmark threshold (`safe_threshold = 0.8 if conf <= 1.0 else 80`).
3. **Adaptive Thresholding (`consensus_engine.py`):** Instead of a static hardcoded threshold, the system employs an Exponential Moving Average (EMA) to track the Maker AI's historical confidence baseline. The safety threshold dynamically adapts (`base_threshold + (alpha * EMA)`). If the primary AI exhibits systemic overconfidence over time, the system automatically tightens the threshold, forcing more cases to the Second-Opinion Checker.
4. **The Second-Opinion AI (Checker):** If the Maker's confidence falls below the dynamic safety threshold, or if high-risk pathology (Grade 3+) is detected, the scan and the Maker's initial hypothesis are routed to `GPT-4o`.
5. **Consensus Resolution:** * If both models agree with high confidence ➡️ **Green: Safe for Auto-Triage**.
   * If there is diagnostic discordance or mutual low confidence ➡️ **Red: Hard Deferral to Human**.

---

## 🛠️ Technical Stack & Implementation Details
* **Frontend:** Streamlit (Optimized for Clinician UI/UX)
* **LLM Orchestration:** Native OpenAI SDK wrapper handling both OpenAI and NVIDIA NIM (Llama) endpoints seamlessly.
* **Resiliency Engineering:** Implemented zero-data-loss checkpointing (atomic OS disk writes) and exponential backoff retry logic (via `Tenacity`) to handle network latency and API rate limits during massive batch processing.
* **Data Pre-processing:** In-memory LANCZOS spatial downsampling via `Pillow` to reduce payload sizes by up to 86% without losing critical biological pathology data (microaneurysms).

---

## 📂 Repository Structure
* `app.py`: The live Streamlit application featuring the Interactive Triage Terminal and Fleet Observability dashboard.
* `pipeline.py`: The production-ready batch processing script used to evaluate the IDRiD cohort.
* `architecture_comparison_report.json`: Telemetry and benchmark data proving the cost/accuracy efficacy of the multi-agent system.
* `consensus_engine.py`: The mathematical routing engine that calculates Adaptive Safety Thresholds (via EMA) and continuous Consensus Risk Scores.
* `results_llama_gpt_images.csv`: The raw audit log of AI discordance.

---

## 🧑‍⚕️ For Clinical Researchers: How to Run a Batch Study

If you are a clinical researcher looking to evaluate a large dataset of patient scans (like a subset of the IDRiD dataset) without writing code, follow this step-by-step guide. 

This pipeline will process thousands of images automatically, handle the AI "Maker-Checker" debate, and output a clean CSV file for your statistical analysis.

### Step 1: Gather Your Prerequisites
Before running the study, you need two things:
1. **Python:** Ensure Python is installed on your computer. 
2. **API Keys (Your Digital ID Badges):** You will need access keys to use the AI models. 
   * Get an [OpenAI API Key](https://platform.openai.com/) (for the GPT-4o Checker).
   * Get an [NVIDIA API Key](https://build.nvidia.com/) (for the Llama-90B Maker).

### Step 2: Prepare Your Data
Create a folder on your computer and place all the retinal scans (`.jpg`, `.png`, or `.tif`) you want to evaluate inside it. 
* *Example:* `C:\Users\DrSmith\Research\Retinal_Scans`

### Step 3: Set Your Environment
Open your computer's Terminal (Mac/Linux) or Command Prompt (Windows) and set your API keys so the script can access them.

**For Mac/Linux:**
```bash
export OPENAI_API_KEY="sk-your-openai-key-here"
export NVIDIA_API_KEY="nvapi-your-nvidia-key-here"
```
**For Windows:**
```bash
set OPENAI_API_KEY="sk-your-openai-key-here"
set NVIDIA_API_KEY="nvapi-your-nvidia-key-here"
```

### Step 4: Configure the Pipeline Script
Open the `pipeline.py` file in a text editor (like Notepad or VS Code).
1.  **Change the Image Path:** Find the line `IMAGE_DIR = "dataset/train_images/`" (usually around line 74).
2.  **Point it to your data:** Change it to the folder where you saved your scans.
    * *Example:* `IMAGE_DIR = "C:\Users\DrSmith\Research\Retinal_Scans"`
3.  **Adjust the Output:** Find `output_path = "result/result_llama_gpt_large.csv"`.
    * *Change it to:* `output_path = "C:\Users\DrSmith\Research\Retinal_Scans\results.csv"`

### Step 5: Run the Evaluation
In your Terminal or Command Prompt (where you set the API keys), run the following command:

```bash
python pipeline.py
```

### Step 6: Analyze the Results
Once the script finishes (it might take a while for a large dataset), it will create a file named `result_llama_gpt_large.csv`.
You can open this file with **Microsoft Excel** or **Google Sheets** to see the detailed breakdown:
* **`scan_name`**: The name of the patient's scan.
* **`model`**: Which AI model made the decision.
* **`model_output`**: The AI's diagnosis (e.g., "No DR", "Mild DR").
* **`correct`**: Whether the AI was right (1) or wrong (0).
* **`image_size`**: How big the file was before the AI analyzed it.
* **`boxes`**: The AI's "reasoning"—the coordinates where it saw the problem.
* **`conf`**: The AI's confidence level (as a percentage).
* **`is_auto_triage`**: Whether the system decided it was safe to proceed automatically (1) or if a doctor needed to review it (0).