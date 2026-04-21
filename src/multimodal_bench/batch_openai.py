import os
import io
import re
import json
import base64
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv()

class OpenAIBatchEvaluator:
    def __init__(self):
        self.client = OpenAI()
        self.images_dir = Path("data/images")
        self.batch_input_file = "batch_input.jsonl"
        self.output_csv = "output.csv"
        
    def _encode_image(self, image_path):
        MAX_DIMENSION = 1024
        COMPRESSION_QUALITY = 85
        
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
                
            img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", optimize=True, quality=COMPRESSION_QUALITY)
            
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
    def prepare_batch(self):
        """Iterates over images, base64 encodes them, and generates batch_input.jsonl"""
        print(f"Preparing batch from {self.images_dir}...")
        if not self.images_dir.exists():
            print(f"Warning: Directory {self.images_dir} does not exist. Creating it.")
            self.images_dir.mkdir(parents=True, exist_ok=True)
            print("Please add your images to this directory.")
            return False

        # 1. Load the Configuration
        config_path = Path("eval_config.json")
        if not config_path.exists():
            print(f"Error: {config_path} not found.")
            return False

        with open(config_path, "r") as f:
            config = json.load(f)
            schema_dict = config.get("schema", {})

        # 2. Construct the System Prompt
        system_prompt = f"""You are an expert ophthalmologist diagnosing Diabetic Retinopathy (DR). 
You must strictly apply the following 0-4 scale:
- 0 (No DR): Absolutely normal retina. No microaneurysms or hemorrhages.
- 1 (Mild): Microaneurysms only.
- 2 (Moderate): More than just microaneurysms, but less than severe. Minor dot-blot hemorrhages.
- 3 (Severe): Any of the following - >20 intraretinal hemorrhages in each of 4 quadrants, definite venous beading in 2+ quadrants, or prominent IRMA in 1+ quadrants.
- 4 (Proliferative): One or both of the following - neovascularization or vitreous/preretinal hemorrhage.

Do not over-diagnose lighting artifacts as hemorrhages.
You must respond in strictly valid JSON matching this exact schema: {json.dumps(config['schema'])}."""

        image_files = [p for p in self.images_dir.glob("*.*") if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']]
        
        if not image_files:
            print(f"No valid images found in {self.images_dir}.")
            return False

        with open(self.batch_input_file, "w") as f:
            for img_path in image_files:
                base64_image = self._encode_image(img_path)
                
                # Standard vision payload Structure
                payload = {
                    "custom_id": f"request-{img_path.name}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o",  # default model
                        "response_format": { "type": "json_object" },
                        "messages": [
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Please analyze this image and provide a JSON response evaluating it."
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        }
                                    }
                                ]
                            }
                        ],
                        "max_tokens": 1000
                    }
                }
                f.write(json.dumps(payload) + "\n")
        print(f"Batch input file created at {self.batch_input_file}")
        return True

    def submit_batch(self):
        """Uploads the batch file and triggers the OpenAI batch job"""
        if not os.path.exists(self.batch_input_file):
            print(f"File {self.batch_input_file} not found. Run prepare_batch() first.")
            return
            
        print("Uploading file to OpenAI...")
        batch_input_file = self.client.files.create(
          file=open(self.batch_input_file, "rb"),
          purpose="batch"
        )
        
        print("Creating batch job...")
        batch_job = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
              "description": "Multimodal bench evaluation"
            }
        )
        
        print(f"Batch job submitted successfully.")
        print(f"Batch ID: {batch_job.id}")
        return batch_job.id

    def retrieve_results(self, batch_id):
        """Checks batch status, downloads result, and parses to output.csv"""
        print(f"Checking status for batch: {batch_id}...")
        batch_job = self.client.batches.retrieve(batch_id)
        
        status = batch_job.status
        print(f"Batch status: {status}")
        
        if status == 'completed':
            output_file_id = batch_job.output_file_id
            if not output_file_id:
                print("No output file found for completed batch.")
                return
                
            print("Downloading results...")
            file_response = self.client.files.content(output_file_id)
            
            output_lines = file_response.text.strip().split('\n')
            results = []
            for line in output_lines:
                if not line:
                    continue
                data = json.loads(line)
                custom_id = data.get('custom_id', '')
                image_filename = custom_id.replace('request-', '')
                
                prediction = "Error"
                confidence_score = "0.0"
                reasoning = "Failed to parse."
                
                try:
                    response_content = data['response']['body']['choices'][0]['message']['content']
                    
                    # Extract JSON using Regex
                    match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_content, re.DOTALL)
                    if match:
                        clean_content = match.group(1).strip()
                    else:
                        # Fallback if no tags are present (raw JSON)
                        clean_content = response_content.strip()
                    
                    parsed_json = json.loads(clean_content)
                    prediction = parsed_json.get('prediction', "Error")
                    confidence_score = parsed_json.get('confidence_score', "0.0")
                    reasoning = parsed_json.get('reasoning', "Failed to parse.")
                    
                except Exception as e:
                    print(f"Warning: Failed to parse response for {image_filename}. Error: {e}")
                    
                results.append({
                    "image_filename": image_filename,
                    "prediction": prediction,
                    "confidence_score": confidence_score,
                    "reasoning": reasoning
                })
                
            df = pd.DataFrame(results)
            df.to_csv(self.output_csv, index=False)
            print(f"Results saved to {self.output_csv}")
        elif status in ['failed', 'expired', 'cancelling', 'cancelled']:
            print(f"Batch failed or cancelled. Details: {batch_job.errors}")
        else:
            print("Batch is still processing. Please try again later.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI Batch API Pipeline")
    parser.add_argument('action', choices=['prepare', 'retrieve'], help="Action to perform")
    parser.add_argument('--batch-id', type=str, help="Batch ID for retrieval", required=False)
    
    args = parser.parse_args()
    
    evaluator = OpenAIBatchEvaluator()
    if args.action == 'prepare':
        if evaluator.prepare_batch():
            evaluator.submit_batch()
    elif args.action == 'retrieve':
        if not args.batch_id:
            print("Error: --batch-id is required for retrieve action")
        else:
            evaluator.retrieve_results(args.batch_id)
