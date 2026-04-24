import gradio as gr
import json
import os
import base64
from io import BytesIO
from openai import OpenAI

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_json(text):
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    except Exception:
        return {}

def call_model(client, model_name, image_b64, prompt):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000,
        temperature=0.0
    )
    return response.choices[0].message.content

def process_image(image, model):
    if image is None:
        empty_clinical = '<div class="clinical-findings">⚪ Waiting for input...</div>'
        empty_banner = '<div class="status-banner">⚪ Waiting for input...</div>'
        return empty_clinical, {}, empty_banner
    
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    nvidia_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ.get("NVIDIA_API_KEY", ""))

    standard_prompt = """Analyze this retinal image. Output a JSON object with:
- "predicted_grade": integer (0-4)
- "confidence_score": integer (0-100)
- "lesion_coordinates": list of objects with "type", "x_coord", "y_coord", "confidence"
Return ONLY valid JSON."""

    reviewer_prompt_template = """You are a Senior Clinical AI Reviewer. A junior model analyzed this retinal image and predicted a clinical grade of {maker_prediction} with these coordinates: {maker_coords}. 
   Review the image critically. 
   You MUST return ONLY a valid JSON object with EXACTLY these three keys, no matter what:
   1. "predicted_grade": Integer strictly from 0 to 4 (0=Healthy, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative).
   2. "confidence_score": Integer strictly from 0 to 100 representing your certainty. Do not use decimals.
   3. "lesion_coordinates": Array of objects. Return [] if none.
   Do not add any other text, markdown, or keys.
   """
    
    image_b64 = encode_image(image)
    
    is_multi_agent = False
    maker_response_text = ""
    checker_response_text = ""
    
    try:
        if model == "OpenAI: GPT-4o-Mini (Fast Screen)":
            maker_response_text = call_model(openai_client, "gpt-4o-mini", image_b64, standard_prompt)
        elif model == "Meta: Llama 3.2 Vision (Open Source)":
            maker_response_text = call_model(nvidia_client, "meta/llama-3.2-90b-vision-instruct", image_b64, standard_prompt)
        elif model == "Alibaba: Qwen 3.5 VLM (High Fidelity)":
            maker_response_text = call_model(nvidia_client, "qwen/qwen3.5-397b-a17b", image_b64, standard_prompt)
        elif model == "Multi-Agent Consensus (OpenAI + Alibaba)":
            is_multi_agent = True
            maker_response_text = call_model(openai_client, "gpt-4o-mini", image_b64, standard_prompt)
            maker_json = extract_json(maker_response_text)
            reviewer_prompt = reviewer_prompt_template.format(
                maker_prediction=maker_json.get("predicted_grade", "None"),
                maker_coords=json.dumps(maker_json.get("lesion_coordinates", []))
            )
            checker_response_text = call_model(nvidia_client, "qwen/qwen3.5-397b-a17b", image_b64, reviewer_prompt)
        elif model == "Multi-Agent Consensus (Meta + Alibaba)":
            is_multi_agent = True
            maker_response_text = call_model(nvidia_client, "meta/llama-3.2-90b-vision-instruct", image_b64, standard_prompt)
            maker_json = extract_json(maker_response_text)
            reviewer_prompt = reviewer_prompt_template.format(
                maker_prediction=maker_json.get("predicted_grade", "None"),
                maker_coords=json.dumps(maker_json.get("lesion_coordinates", []))
            )
            checker_response_text = call_model(nvidia_client, "qwen/qwen3.5-397b-a17b", image_b64, reviewer_prompt)
    except Exception as e:
        return "Error", "Error", f"### Status: 🔴 Error: {str(e)}", {}
        
    final_response_text = checker_response_text if is_multi_agent else maker_response_text
    final_json = extract_json(final_response_text)
    
    predicted_grade = str(final_json.get("predicted_grade", "N/A"))
    confidence_score = final_json.get("confidence_score", "N/A")
    lesion_coordinates = final_json.get("lesion_coordinates", [])
    
    # Map 0-4 Grade
    grade_map = {
        "0": "Healthy (No DR)",
        "1": "Mild DR",
        "2": "Moderate DR",
        "3": "Severe DR",
        "4": "Proliferative DR"
    }
    mapped_grade = grade_map.get(predicted_grade, predicted_grade)

    # 2. Clinical Summary HTML
    clinical_summary_html = f"""
    <div class="clinical-findings">
        <div class="clinical-grade">Predicted Clinical Grade: {predicted_grade} - {mapped_grade}</div>
        <div class="confidence">Confidence Score: {confidence_score}%</div>
    </div>
    """

    # 3. Python dictionary for JSON
    lesion_dict = {"lesion_coordinates": lesion_coordinates}

    # 4. Status Banner
    status_banner_html = ""
    try:
        conf = int(confidence_score)
        if conf >= 85:
            status_banner_html = '<div class="status-banner approved">🟢 Routine Queue: AI Consensus Achieved</div>'
        else:
            status_banner_html = '<div class="status-banner review">🔴 High Priority: Attending Physician Review Required</div>'
    except:
        status_banner_html = '<div class="status-banner review">🔴 Status Unknown</div>'

    return clinical_summary_html, lesion_dict, status_banner_html

css = """
.status-banner { padding: 15px; border-radius: 8px; font-weight: bold; font-size: 16px; margin-top: 10px; }
.approved { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
.review { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
.clinical-findings { padding: 15px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6; margin-bottom: 10px; }
.clinical-grade { font-size: 24px; font-weight: bold; color: #007bff; }
.confidence { font-size: 18px; color: #495057; }
.agentic-flow { padding: 10px; background-color: #e2e3e5; border-radius: 8px; font-weight: bold; margin-bottom: 15px; }
.override { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
"""

# Build the Gradio UI
with gr.Blocks(title="MedVision-Bench Playground", theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("# MedVision AI: Clinical Second Opinion Dashboard")
    gr.Markdown("Upload a patient's Retinal Fundus scan below. The AI will evaluate the image for Diabetic Retinopathy and flag complex cases for physician review.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Patient Retinal Scan (Upload JPEG/PNG)")
            model_dropdown = gr.Dropdown(
                choices=[
                    "OpenAI: GPT-4o-Mini (Fast Screen)", 
                    "Meta: Llama 3.2 Vision (Open Source)", 
                    "Alibaba: Qwen 3.5 VLM (High Fidelity)", 
                    "Multi-Agent Consensus (OpenAI + Alibaba)", 
                    "Multi-Agent Consensus (Meta + Alibaba)"
                ],
                value="OpenAI: GPT-4o-Mini (Fast Screen)",
                label="Select Diagnostic Protocol"
            )
            submit_btn = gr.Button("Analyze Scan & Generate Second Opinion", variant="primary")
            
        with gr.Column():
            status_banner_output = gr.HTML('<div class="status-banner">⚪ Waiting for input...</div>')
            
            with gr.Group():
                clinical_output = gr.HTML('<div class="clinical-findings">⚪ Waiting for input...</div>')
                with gr.Accordion("🔍 View AI Spatial Coordinates (Audit Trail)", open=False):
                    lesion_output = gr.JSON(label="Raw AI Bounding Boxes (Coordinates)", value={})

    # Wire up the logic
    submit_btn.click(
        fn=process_image,
        inputs=[image_input, model_dropdown],
        outputs=[clinical_output, lesion_output, status_banner_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
