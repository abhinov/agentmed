"""
AgenticMed Core Evaluation Pipeline
Benchmarks multimodal LLMs against high-resolution ophthalmic images.
"""

import os
import io
import csv
import json
import base64
import argparse
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# ==========================================
# CONFIGURATION & SCHEMA
# ==========================================
OUTPUT_FILE = "results.csv"
FIELDNAMES = [
    'filename', 
    'diagnosis', 
    'confidence_score', 
    'clinical_triage_plan', 
    'patient_explanation',
    'error'
]

# Strict JSON Schema Enforcement (Architectural Guardrail 4)
SYSTEM_PROMPT = """You are an expert ophthalmic AI triage system. 

You MUST output your response as pure JSON matching this exact schema:
{
    "diagnosis": "string",
    "confidence_score": integer (0-100),
    "clinical_triage_plan": "string",
    "patient_explanation": "string"
}"""

# ==========================================
# LOCAL PRE-PROCESSING
# ==========================================
def encode_image_local(image_path: Path) -> str:
    """
    Architectural Guardrail 1: Local Pre-Processing (Cost/Memory)
    Safely opens, resizes (max 1024x1024 maintaining aspect ratio), 
    converts to RGB, and base64 encodes the image.
    Raises UnidentifiedImageError for corrupted files to be caught in the main loop.
    """
    MAX_DIMENSION = 1024
    COMPRESSION_QUALITY = 85
    
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        # thumbnail modifies in-place and preserves aspect ratio
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        # optimize=True provides further lossless reduction for JPEG
        img.save(buffer, format="JPEG", optimize=True, quality=COMPRESSION_QUALITY)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

# ==========================================
# STATE CHECKPOINTING
# ==========================================
def load_processed_state(output_file: str) -> set:
    """
    Architectural Guardrail 2: State Checkpointing (Fault Tolerance)
    Reads the existing results.csv to build a Set of already processed filenames.
    """
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('filename'):
                    processed.add(row['filename'])
    return processed

def init_output_file(output_file: str, fieldnames: list):
    """Initializes the output CSV with headers if it doesn't exist."""
    if not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

def append_result(output_file: str, fieldnames: list, result: dict):
    """Appends a single row to the output CSV."""
    with open(output_file, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(result)

def safe_parse_json(response_text: str) -> dict:
    """Safely extracts JSON from Markdown-fenced or raw response strings."""
    try:
        import re
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
        clean_content = match.group(1).strip() if match else response_text.strip()
        return json.loads(clean_content)
    except Exception:
        return {}

# ==========================================
# MODULAR API ROUTING & RESILIENCY
# ==========================================
# Architectural Guardrail 3: Resilient API Calls using tenacity exponential backoff

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception) # Safely catches 429s/500s broadly
)
def call_openai_api(image_b64: str) -> str:
    """Architectural Guardrail 6: Modular OpenAI endpoint execution."""
    # TODO: Insert API key here or rely on environment variables
    # Example generic invocation:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Evaluate this retinal scan outputting strictly valid JSON."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }
        ],
        max_tokens=1000,
        temperature=0.0
    )
    return response.choices[0].message.content

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception)
)
def call_gemini_api(image_b64: str) -> str:
    """Architectural Guardrail 6: Modular Gemini endpoint execution."""
    # TODO: Insert API key here or rely on environment variables
    # Implement google-genai or vertexai SDK logic here
    raise NotImplementedError("Gemini SDK integration required.")

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception)
)
def call_grok_api(image_b64: str) -> str:
    """Architectural Guardrail 6: Modular Grok endpoint execution."""
    # TODO: Insert API key here or rely on environment variables
    raise NotImplementedError("Grok SDK integration required.")

# ==========================================
# MAIN EXECUTION LOOP
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Multimodal Bench Evaluation Pipeline")
    parser.add_argument("--model", type=str, choices=["openai", "gemini", "grok"], default="openai", help="Target model router")
    args = parser.parse_args()

    images_dir = Path("data/images")
    if not images_dir.exists():
        print(f"Error: Directory {images_dir} does not exist.")
        return

    # Gather target images
    image_files = [p for p in images_dir.glob("*.*") if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']]
    if not image_files:
        print(f"No valid images found in {images_dir}.")
        return

    # Initialize Checkpointing Protocol
    init_output_file(OUTPUT_FILE, FIELDNAMES)
    processed_files = load_processed_state(OUTPUT_FILE)
    
    # Filter unprocessed images
    to_process = [img for img in image_files if img.name not in processed_files]
    print(f"Total images: {len(image_files)} | Processed: {len(processed_files)} | Remaining: {len(to_process)}")

    # Map the targeted API router function
    api_router = {
        "openai": call_openai_api,
        "gemini": call_gemini_api,
        "grok": call_grok_api
    }[args.model]

    # Architectural Guardrail 5: Progress Tracking
    for img_path in tqdm(to_process, desc=f"Evaluating with {args.model}"):
        
        result_dict = {}
        error_msg = ""
        
        try:
            # Safely catch corrupted images inside the loop
            base64_image = encode_image_local(img_path)
            
            # Execute routed API with embedded resiliency
            response_text = api_router(base64_image)
            result_dict = safe_parse_json(response_text)
            
        except UnidentifiedImageError:
            error_msg = "UnidentifiedImageError: Image corrupted or invalid format."
        except Exception as e:
            error_msg = str(e)
            
        # Compile result row ensuring enforced schema fallback values
        row = {
            'filename': img_path.name,
            'diagnosis': result_dict.get('diagnosis', 'Error'),
            'confidence_score': result_dict.get('confidence_score', 0),
            'clinical_triage_plan': result_dict.get('clinical_triage_plan', 'N/A'),
            'patient_explanation': result_dict.get('patient_explanation', 'N/A'),
            'error': error_msg
        }
        
        # Write state instantly to disk
        append_result(OUTPUT_FILE, FIELDNAMES, row)

if __name__ == "__main__":
    main()
