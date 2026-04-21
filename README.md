# Multimodal Bench: AI Vision Evaluator

A simple, automated tool that lets you test how well AI models (like ChatGPT) can analyze your image datasets. You bring the images and the grading rubric, and this tool does the rest.

## Step 1: Download & Setup (No coding required)

1. First, download the tool to your computer:
   ```bash
   git clone https://github.com/abhinov/multimodal-bench.git
   cd multimodal-bench
   ```

2. Install the required tools:
   ```bash
   pip install -r requirements.txt
   ```

3. **Secure Your API Key:** 
   Rename the `.env.example` file to `.env`. Open it and paste your OpenAI API key inside (e.g. `OPENAI_API_KEY="sk-..."`). This file is hidden and completely secure—your key will never be uploaded anywhere.

## Step 2: Prepare Your Data

- **Images:** Place all of the images you want the AI to analyze into the `data/images/` folder.
- **Answer Key:** To let the tool know how to grade the AI, simply open `scripts/format_labels.py`. Put the name of your Excel/CSV file at the top of the script, and then run:
  ```bash
  python scripts/format_labels.py
  ```
  This will automatically format your answer key!

## Step 3: Tell the AI What to Look For

The `eval_config.json` file is essentially your "Rubric". Open it to define exactly what instructions to give the AI and which model to use. 

Here is a simple example:
```json
{
  "model": "gpt-4o-mini",
  "prompt": "You are a medical expert. Look closely at this image and diagnose the condition.",
  "expected_format": "Select one: [Healthy, Mild, Severe]"
}
```
*Note: You can easily change `"gpt-4o-mini"` to whichever model you'd like to test!*

## Step 4: Run the Analysis

You have two ways to run your analysis, depending on your needs:

### Option A (Fast Test)
Want to test a few images immediately to see how it works? Run:
```bash
make run-openai
```

### Option B (Massive Datasets - 50% Cheaper)
If you are running thousands of images, use the Batch API to get a massive discount! Kick off the process by running:
```bash
make prepare-batch
```
The console will give you a `BATCH_ID`. Wait a few hours for OpenAI to process the images, then retrieve your results by running:
```bash
make retrieve-batch BATCH_ID="your_id_here"
```

## Step 5: Get Your Grade

Once the AI has finished answering, you can grade it! Simply run:
```bash
make grade
```
This command will compare all of the AI's answers to your answer key and give you a final accuracy score for your dataset.

## 📏 The Five Enterprise Evaluation Parameters

- **1. Clinical Accuracy:** Exact match percentage against the 0-4 clinical grading scale.
- **2. False Positive Rate (The "Overthinking Penalty"):** The rate at which the model hallucinates disease on perfectly healthy retinas.
- **3. Unit Economics:** The exact API cost to evaluate 1,000 images using Asynchronous Batching.
- **4. Architecture Resilience:** The accuracy delta between Zero-Shot prompting and Calibrated Multi-Shot prompting.
- **5. Visual Grounding:** The model's sensitivity to payload compression (e.g., 57KB vs 1024px fidelity).
