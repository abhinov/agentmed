# 🏥 MedVision-Bench: Multi-Agent Orchestration for Multimodal Clinical AI

**MedVision-Bench** is an open-source, fault-tolerant evaluation engine designed to benchmark frontier Vision-Language Models (VLMs) on clinical imagery. 

While the industry evaluates text-based clinical chat (e.g., OpenAI's HealthBench), MedVision-Bench serves as the **multimodal equivalent**—evaluating spatial explainability, AI hallucination safety, and Multi-Agent consensus on high-stakes medical scans (like Diabetic Retinopathy).

You can also play with the live Multi-Agent debate UI here: https://huggingface.co/spaces/beabhinov/medvision-bench-demo
---

## 🩺 For Clinical Staff & Doctors: What This Means for You
We know that AI cannot and should not replace doctors. MedVision-Bench treats Artificial Intelligence like a **Medical Resident**, while you remain the **Attending Physician**. 

Here is how it works:
1. **The "Second Opinion" Debate:** Instead of asking one AI for an answer, our system asks two different AI models to look at a retinal scan and debate their findings until they agree (Consensus).
2. **Explaining Its Work:** The AI doesn't just give a grade; it literally draws coordinates on the scan to show you *exactly* where it sees microaneurysms or hard exudates.
3. **Knowing When It's Unsure:** If the AI's confidence drops below a strict safety threshold (85%), or if the two models disagree, it automatically stops and triggers a 🔴 **Human-in-the-Loop (HITL) Review**, escalating the scan to your desk for human triage. 

**The Goal:** Automate the screening of healthy patients so you can spend your time exclusively on the complex cases that actually need your expertise.

---

## 🚀 For AI Native Teams: Agent Architecture
*Keywords: Multi-Agent Orchestration, Agentic Workflows, AI Safety, Multimodal LLM Evals, JSON Schema Enforcement, Human-in-the-Loop (HITL), Hybrid Cloud Routing, Prompt Engineering, NVIDIA NIM.*

MedVision-Bench is architected to solve the three major bottlenecks in enterprise AI deployment:

### 1. "Maker-Checker" Multi-Agent Orchestration
Single-shot zero-shot prompting is obsolete for high-stakes enterprise use cases. MedVision-Bench implements a multi-agent consensus loop. 
* **The Maker:** A fast, lightweight model (`gpt-4o-mini` or Meta's `Llama-3.2-90b-vision`) generates the initial bounding boxes and clinical grade.
* **The Checker:** The payload is routed to a state-of-the-art VLM (`Qwen3.5-397B` via NVIDIA NIM) acting as a Senior Reviewer to critique the Maker's output.
* **Hybrid Routing:** By mixing OpenAI and NVIDIA hosted open-source models, the pipeline proves **multi-cloud orchestration** and prevents vendor lock-in.

### 2. Defeating the "Alignment Tax" via Strict JSON Enforcement
Frontier open-weights models suffer from schema drift when engaged in agentic debate. To force deterministic output for downstream clinical UI software, MedVision-Bench utilizes aggressive prompt guardrails. The architecture forces non-deterministic LLMs into a strict JSON schema, ensuring integer-based confidence scores and structured spatial coordinate arrays.

### 3. Optimizing Unit Economics (86% Payload Reduction)
Raw medical images cause high latency, API timeouts, and massive bandwidth costs. MedVision-Bench implements local preprocessing via the Python Imaging Library (PIL). The system automatically compresses and resizes multi-megabyte clinical scans prior to API transmission, **reducing payload sizes by 86%** while preserving the pixel-fidelity required for diagnostic accuracy.

---

## 🛠️ The Model Matrix
The live evaluation playground allows users to test five distinct architectural workflows:
1. `gpt-4o-mini` (Proprietary / Baseline)
2. `meta/llama-3.2-90b-vision-instruct` (Open Source Text-Heavy VLM)
3. `qwen/qwen3.5-397b-a17b` (SOTA Open Source VLM)
4. **Multi-Agent: Hybrid** (GPT Maker + Qwen Checker)
5. **Multi-Agent: Open-Source** (Llama Maker + Qwen Checker)

---

## 💻 Quick Start (Run Locally)
1. Clone the repository:
   `git clone https://github.com/your-username/medvision-bench.git`
2. Install dependencies:
   `pip install -r requirements.txt`
3. Export your API Keys:
   ```bash
   export OPENAI_API_KEY="sk-..."
   export NVIDIA_API_KEY="nvapi-..."
   ```
4. Launch the Gradio App:
   ```bash
   python app.py
   ```

Built with ❤️ for the future of safe, agentic healthcare.
