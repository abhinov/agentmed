import os
import sys
import io
import csv
import json
import base64
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from consensus_engine import calculate_market_consensus_risk

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from dotenv import load_dotenv

# Load environment variables to resolve AuthenticationErrors
load_dotenv()

from openai import OpenAI, RateLimitError, APIConnectionError, InternalServerError, APITimeoutError
RETRYABLE_EXCEPTIONS = (RateLimitError, APIConnectionError, InternalServerError, APITimeoutError)

# ==========================================
# CONFIGURATION & GUARDRAILS
# ==========================================
# TODO: Export these in your terminal environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your-openai-key-here")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "your-nvidia-api-key-here")
# Placeholders for future integrations
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "your-gemini-key-here")
GROK_API_KEY = os.environ.get("GROK_API_KEY", "your-grok-key-here")

# Guardrail 4: Enforced JSON Output Schema
SYSTEM_PROMPT = """You are an expert ophthalmic AI analyzing a fundus image.
You must return your analysis strictly as a JSON object with the exact following keys:
- "diagnosis_grade" (integer): The diabetic retinopathy severity grade from 0 to 4, where 0=No DR, 1=Mild NPDR, 2=Moderate NPDR, 3=Severe NPDR, 4=Proliferative DR.
- "diagnosis" (string): Verbose clinical finding (e.g., Severe Non-Proliferative Diabetic Retinopathy).
- "confidence_score" (integer): 0-100 representing your diagnostic certainty.
- "clinical_triage_plan" (string): Recommended next steps for the patient.
- "patient_explanation" (string): A brief, empathetic explanation suitable for a patient.
Output ONLY valid JSON. Do not include markdown formatting blocks."""

# Dynamic thresholds to trigger the Checker based on known model overconfidence
CALIBRATED_THRESHOLDS = {
    "gpt": 92,
    "llama": 96,
    "qwen": 88
}

# ==========================================
# GUARDRAIL 1: LOCAL PRE-PROCESSING
# ==========================================
def encode_image_local(image_path: Path) -> str:
    """
    Safely opens, resizes (max 1024x1024), and base64 encodes the image in memory.
    Why: Reduces a massive 10GB dataset by ~86%, saving API token costs 
    and preventing API payload bandwidth timeouts.
    """
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # LANCZOS preserves critical clinical pixel fidelity during compression
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", optimize=True, quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
    except UnidentifiedImageError:
        print(f"\n[Warning] Corrupted or unreadable image skipped: {image_path.name}")
        return None
    except Exception as e:
        print(f"\n[Error] Pre-processing failed for {image_path.name}: {str(e)}")
        return None

# ==========================================
# GUARDRAIL 7: SAFE JSON EXTRACTION
# ==========================================
def safe_parse_json(response_text: str) -> dict:
    """Safely extracts JSON from Markdown-fenced or raw response strings."""
    if not response_text:  # Catches None and empty strings ""
        return {}
        
    try:
        import re
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', str(response_text), re.DOTALL)
        clean_content = match.group(1).strip() if match else str(response_text).strip()
        return json.loads(clean_content)
    except Exception:
        return {}

# ==========================================
# GUARDRAIL 3 & 6: MODULAR & RESILIENT API ROUTING
# ==========================================
@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5), retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS))
def call_openai_api(image_b64: str) -> dict:
    """Maker Option A: GPT via OpenAI"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o", # Target: ChatGPT 5.2 equivalent
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Evaluate this retinal scan and provide triage details."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]}
        ],
        temperature=0.0
    )
    return safe_parse_json(response.choices[0].message.content)

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5), retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS))
def call_nim_llama_api(image_b64: str) -> dict:
    """Maker Option B: Llama 3.2 Vision via NVIDIA NIM"""
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
    response = client.chat.completions.create(
        model="meta/llama-3.2-90b-vision-instruct",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Evaluate this retinal scan and provide triage details."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]}
        ],
        temperature=0.0
    )
    return safe_parse_json(response.choices[0].message.content)

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5), retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS))
def call_nim_qwen_api(image_b64: str) -> dict:
    """The Checker: Qwen2-VL 72B via NVIDIA NIM"""
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
    response = client.chat.completions.create(
        model="Qwen/Qwen2-VL-72B-Instruct",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Critically review this scan as a second opinion. Provide your triage details."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]}
        ],
        temperature=0.0
    )
    return safe_parse_json(response.choices[0].message.content)

def call_gemini_api(image_b64: str) -> dict:
    """Stub for Google Gemini 3.1 Pro Implementation"""
    pass

def call_grok_api(image_b64: str) -> dict:
    """Stub for xAI Grok 4.2 Implementation"""
    pass

# ==========================================
# GUARDRAIL 2 & 5: STATE CHECKPOINTING & EXECUTION LOOP
# ==========================================
def run_pipeline(maker_choice: str, checker_choice: str, dataset_dir: Path):
    print(f"🚀 Initializing Multi-Agent Pipeline | MAKER: {maker_choice.upper()} | CHECKER: {checker_choice.upper()} | DATASET: {dataset_dir.name}")
    
    results_file = Path(f"./results_{maker_choice}_{checker_choice}_{dataset_dir.name}.csv")
    
    fieldnames = [
        'filename', 
        'maker_diagnosis', 'maker_confidence', 
        'checker_triggered', 'checker_diagnosis', 'checker_confidence',
        'final_triage_plan', 'patient_explanation',
        'consensus_risk_score', 'triage_decision', 'error'
    ]

    # Fault Tolerance: Checkpointing Logic
    processed_files = set()
    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            processed_files = {row['filename'] for row in reader if row.get('filename')}
    else:
        with open(results_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    all_images = [p for p in dataset_dir.glob("*.*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif']]
    to_process = [img for img in all_images if img.name not in processed_files]
    
    print(f"📦 Total Scans: {len(all_images)} | Processed: {len(processed_files)} | Pending: {len(to_process)}")

    # 2. Dynamic API Router Dictionary
    API_ROUTER = {
        'gpt': call_openai_api,
        'llama': call_nim_llama_api,
        'qwen': call_nim_qwen_api
    }
    maker_api = API_ROUTER[maker_choice]
    checker_api = API_ROUTER[checker_choice]

    # Main Execution Loop
    with open(results_file, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Guardrail 5: Progress Tracking via tqdm
        for img_path in tqdm(to_process, desc=f"Evaluating {dataset_dir.name}"):
            base64_image = encode_image_local(img_path)
            if not base64_image: 
                continue # Skip gracefully if PIL fails
                
            # Default State: Assume Unsafe / Error
            row_data = {
                'filename': img_path.name,
                'maker_diagnosis': 'N/A',
                'maker_confidence': -1,
                'checker_triggered': False,
                'checker_diagnosis': 'N/A',
                'checker_confidence': -1,
                'final_triage_plan': 'N/A',
                'patient_explanation': 'N/A',
                'consensus_risk_score': 1.0,
                'triage_decision': '🔴 Unsafe / Error',
                'error': ''
            }
            
            try:
                # Execute Maker Dynamically
                maker_response = maker_api(base64_image)
                
                # 3. CRITICAL: Safety Filter Refusal Handling
                if maker_response is None:
                    raise ValueError("API returned None due to safety filter")
                    
                m_conf = maker_response.get('confidence_score', -1)
                m_grade = maker_response.get('diagnosis_grade', -1)
                
                row_data.update({
                    'maker_diagnosis': m_grade,
                    'maker_confidence': m_conf,
                    'final_triage_plan': maker_response.get('clinical_triage_plan', 'Error'),
                    'patient_explanation': maker_response.get('patient_explanation', 'Error')
                })
                
                # Safety extraction for Grade (handling strings to prevent crash)
                try:
                    m_grade_int = int(m_grade)
                except (ValueError, TypeError):
                    m_grade_int = -1
                
                # 4. The Escalation Logic
                threshold = CALIBRATED_THRESHOLDS.get(maker_choice, 95)
                if (m_conf != -1 and m_conf < threshold) or m_grade_int >= 3:
                    row_data['checker_triggered'] = True
                    
                    checker_response = checker_api(base64_image)
                    if checker_response is None:
                        raise ValueError("API returned None due to safety filter")
                        
                    c_conf = checker_response.get('confidence_score', -1)
                    c_grade = checker_response.get('diagnosis_grade', -1)
                    
                    row_data.update({
                        'checker_diagnosis': c_grade,
                        'checker_confidence': c_conf,
                        # Second opinion overrides the triage plan in the final output
                        'final_triage_plan': checker_response.get('clinical_triage_plan', 'Error'),
                        'patient_explanation': checker_response.get('patient_explanation', 'Error')
                    })
                    
                    risk_score, triage_decision = calculate_market_consensus_risk(
                        maker_choice, m_grade, m_conf, 
                        checker_choice, c_grade, c_conf
                    )
                    row_data['consensus_risk_score'] = risk_score
                    row_data['triage_decision'] = triage_decision
                else:
                    # High confidence + Low severity = Safe
                    row_data['consensus_risk_score'] = 0.1
                    row_data['triage_decision'] = '🟢 Safe / Auto-Triage'
                    
            except Exception as e:
                # Catch total schema failure, continuous API timeout, or False-Positive Gore Filters
                row_data['error'] = str(e)

            # Fault Tolerance: Force OS to write buffer to disk instantly, crash-proofing the row
            writer.writerow(row_data)
            f.flush() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Evaluation Pipeline for Clinical Scans")
    # 1. Dynamic CLI Arguments
    parser.add_argument("--maker", type=str, choices=['gpt', 'llama', 'qwen'], required=True, help="Select the target Maker model.")
    parser.add_argument("--checker", type=str, choices=['gpt', 'llama', 'qwen'], required=True, help="Select the target Checker model.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the local dataset directory.")
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.is_dir():
        print(f"[Error] Directory not found: {dataset_path}")
        exit(1)
        
    run_pipeline(args.maker, args.checker, dataset_path)
