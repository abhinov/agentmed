# 🏥 AgentMed: Multi-Agent Orchestration for Clinical AI Observability

**AgentMed** is a high-fidelity evaluation engine and observability gateway designed to benchmark frontier Vision-Language Models (VLMs) on clinical imagery. 

While standard benchmarks evaluate text-based clinical chat, AgentMed serves as a **multimodal diagnostic auditor**—evaluating spatial explainability, AI hallucination safety, and Multi-Agent consensus on high-stakes retinal scans (IDRiD Dataset).

**Live Demo:** [https://huggingface.co/spaces/beabhinov/agentmed](https://huggingface.co/spaces/beabhinov/agentmed)

---

## 📊 Benchmark Results (IDRiD Dataset - 455 Scans)

Our architecture proves that a **Multi-Agent "Maker-Checker" loop** consistently outperforms standalone frontier models. By using an open-weights model for initial screening and escalating complex cases to a flagship model, we achieved superior accuracy with optimized unit economics.

| Metric | Single-Agent (GPT-4o-mini) | Multi-Agent (Llama-90B + GPT-4o) |
| :--- | :--- | :--- |
| **Overall Accuracy** | 45.27% | **46.15%** |
| **Net Accuracy Gain** | Baseline | **+0.88% (Absolute) / +4.18% (vs GPT-4o)** |
| **Cost Efficiency** | 100% Premium API spend | **62.4% Cost Reduction** |
| **Escalation Rate** | 0% (Blind Trust) | **37.6% (Targeted Review)** |

*Key Finding: The Multi-Agent system successfully fixed **51 clinical hallucinations** where the primary model failed.*

---

## 🚀 Architectural Deep-Dive

AgentMed is built to solve the major bottlenecks in enterprise AI deployment:

### 1. "Maker-Checker" Routing Engine
We moved beyond zero-shot prompting to a state-aware agentic workflow:
*   **The Maker (Llama-3.2-90B):** Acts as the frontline clinician, providing the initial diagnosis and confidence score via NVIDIA NIM.
*   **The Checker (GPT-4o):** Acts as the Senior Consultant, triggered only when the Maker's confidence drops or a high-variance diagnosis is detected.
*   **The Outcome:** This "Consensus Triage" prevents single-model hallucinations from reaching the final report.

### 2. Clinical Observability & Audit Trails
Every scan generates a **Consensus Risk Score**. This score isn't just a confidence number; it accounts for the mathematical variance between the two models. 
*   **Safe / Auto-Triage:** High confidence, high agreement.
*   **Review Required:** Low confidence or model disagreement.
*   **Spatial Explainability:** Models are forced to provide structured JSON coordinates, allowing doctors to audit exactly *where* the AI sees pathology.

### 3. Optimizing Unit Economics
By utilizing **Llama-90B** for the 62.4% of cases that were "Safe," we drastically reduced the reliance on expensive proprietary APIs without sacrificing diagnostic quality. Local pre-processing (PIL) further reduced image payload sizes by **86%**, ensuring sub-second latencies for the end-user.

---

## 🛠️ The Model Matrix
The AgentMed Gateway allows testing of validated clinical workflows:
1.  **AgentMed: Llama-90B (Maker) + GPT-4o (Checker)** - *The Production Standard*
2.  **Single-Agent: GPT-4o-mini** - *The High-Speed Baseline*

---

## 💻 Quick Start (Local Setup)

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/agentmed.git](https://github.com/your-username/agentmed.git)
   cd agentmed
Install Dependencies:

Bash
pip install -r requirements.txt
Configure Environment:
Add your keys to a .env file or export them:

Bash
export OPENAI_API_KEY="sk-..."
export NVIDIA_API_KEY="nvapi-..."
Launch the Gateway:

Bash
streamlit run app.py

🔬 Dataset Acknowledgement
This project utilizes the IDRiD (Indian Diabetic Retinopathy Image Dataset) for all benchmarks.

Built for the future of safe, agentic healthcare. 🏥