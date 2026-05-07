# 🏥 AgentMed: Clinical AI Copilot & Multi-Agent Orchestrator

**AgentMed** is a reliable clinical AI Copilot. It acts as an automated "second opinion" system, testing how well modern AI models analyze complex medical images like retinal scans. 

Instead of relying on a single AI, AgentMed uses a **"Maker-Checker" system**—where multiple AI models debate a diagnosis to ensure accuracy, safety, and cost-efficiency before handing the final decision to a human doctor.

**Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/beabhinov/agentmed)

> **⚠️ Note on the Live Demo:** To make sure you see the AI debate in action, the web demo **forces** the Second-Opinion AI to review every single scan. It also uses a simple, fixed rule to decide if a scan is safe. To see our smart, cost-saving routing in action, check out the main production code (`pipeline.py` and `consensus_engine.py`).

---

## 🚀 Business Impact & Product Value
Building AI for healthcare usually forces a tough choice: use cheap models that might make mistakes, or use expensive, massive models that cost too much to run at scale. AgentMed solves this.

### 1. Better Accuracy Through AI Teamwork
By forcing two different AI models to double-check ambiguous scans, AgentMed is significantly more accurate than using a single AI.
* **Baseline (Single AI - GPT-4o-mini):** 66.59% Accuracy
* **AgentMed Pipeline (Two AIs debating):** **72.31% Accuracy** (+5.72% uplift)

### 2. Lowering Costs (Doing More with Less)
Even though we use a premium, expensive model (GPT-4o) for second opinions, our overall system is **cheaper** than running a standard model across all data. 
* **Baseline Cost:** $0.29 per scan
* **AgentMed Cost:** **$0.27 per scan**

**How?** We use a powerful, free open-source model (Llama-3.2-90B) to do the heavy lifting on the easiest cases. The expensive premium model (GPT-4o) is only triggered when the first AI is unsure or detects a high-risk disease. 

### 3. Smart Risk Management (Self-Adjusting Rules)
Instead of hardcoding rules, AgentMed learns over time. If the frontline AI starts acting overly confident about difficult scans, the system automatically tightens its safety net, forcing more scans to get a second opinion. If the AI proves reliable, the system relaxes the rules to save money.

### 4. Keeping Humans in Control
AgentMed treats AI like a **Medical Resident**, while the human user remains the **Attending Physician**. 
* **Safe Auto-Triage Rate (62.42%):** Cases where the AIs agreed with high confidence.
* **Human Escalation Rate (37.58%):** Cases where the AIs disagreed or were unsure. These are flagged as a "Hard Deferral" and immediately sent to a human doctor.

---

## 🧠 System Architecture

This architecture proves that having AI models check each other's work drastically reduces errors and "hallucinations" in medical tasks. 

1. **The Primary AI (Maker):** Meta's open-source `Llama-3.2` looks at the scan first, offering a diagnosis and a confidence score (e.g., "I am 85% sure this is Grade 2").
2. **The Second-Opinion AI (Checker):** If the Maker's confidence is too low, or if it spots a severe disease, the scan is automatically sent to `GPT-4o-mini` for a strict audit.
3. **Adaptive Safety Net (`consensus_engine.py`):** The system constantly tracks the Maker AI's track record. If the Maker's historical confidence fluctuates, the system dynamically changes the minimum score required to pass without a second opinion. 
4. **Final Decision:** * If both AIs agree and are highly confident ➡️ **Green: Safe for Auto-Triage**.
   * If the AIs disagree or are guessing ➡️ **Red: Hard Deferral to a Human Doctor**.

---

## 🛠️ Technical Stack & Implementation Details
* **Frontend:** Built with Streamlit, optimized to look and feel like a real doctor's dashboard.
* **Dynamic LLM Orchestration:** A unified API wrapper interfaces seamlessly with both OpenAI (GPT) and NVIDIA (Llama). It utilizes a dynamic Maker-Checker flow that intelligently routes scans to a heavier, secondary AI only when necessary..
* **Fault-Tolerant Batch Processing:** Built with auto-saving and exponential backoff retry loops. This ensures zero data loss and prevents the pipeline from crashing during network latency or API rate limits.
* **Smart Image Compression:** Resizes high-resolution biological data in-memory, shrinking file sizes by up to 86% without blurring out tiny, critical details like microaneurysms.

---

## 📂 Repository Structure
* `app.py`: The live application featuring the Interactive Patient Triage Terminal and Fleet Observability dashboard.
* `pipeline.py`: The main processing script used to process large batches of images.
* `architecture_comparison_report.json`: Telemetry and benchmark data proving the cost/accuracy efficacy of the multi-agent system.
* `consensus_engine.py`: (Smart Safety Net) The brain behind the AI debate. Instead of fixed rules, it constantly monitors the primary AI's track record. If the primary AI becomes too confident on tricky scans, the system tightens the rules, forcing more cases to get a secondary AI opinion. This self-adjusting safety net guarantees patient safety while optimizing costs.
* `results_llama_gpt_images.csv`: The raw logs of every time the AIs debated a scan.

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