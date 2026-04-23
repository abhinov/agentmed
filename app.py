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
        return "N/A", "N/A", "### Status: ⚪ Waiting for input...", "{}"
    
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
        if model == "gpt-4o-mini":
            maker_response_text = call_model(openai_client, model, image_b64, standard_prompt)
        elif model in ["meta/llama-3.2-90b-vision-instruct", "qwen/qwen3.5-397b-a17b"]:
            maker_response_text = call_model(nvidia_client, model, image_b64, standard_prompt)
        elif model == "Multi-Agent: Hybrid (GPT Maker + Qwen Checker)":
            is_multi_agent = True
            maker_response_text = call_model(openai_client, "gpt-4o-mini", image_b64, standard_prompt)
            maker_json = extract_json(maker_response_text)
            reviewer_prompt = reviewer_prompt_template.format(
                maker_prediction=maker_json.get("predicted_grade", "None"),
                maker_coords=json.dumps(maker_json.get("lesion_coordinates", []))
            )
            checker_response_text = call_model(nvidia_client, "qwen/qwen3.5-397b-a17b", image_b64, reviewer_prompt)
        elif model == "Multi-Agent: Open-Source (Llama Maker + Qwen Checker)":
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
    
    status_banner = ""
    try:
        conf = int(confidence_score)
        if conf >= 85:
            status_banner = "### Status: 🟢 Auto-Triage Approved"
        else:
            status_banner = "### Status: 🔴 Human-in-the-Loop Review Required"
    except:
        status_banner = "### Status: 🔴 Status Unknown"
        
    if is_multi_agent:
        maker_json = extract_json(maker_response_text)
        maker_grade = str(maker_json.get("predicted_grade", "N/A"))
        if maker_grade != predicted_grade:
            status_banner += "\n\n🟡 Reviewer Overrode Maker"
            
    return predicted_grade, str(confidence_score), status_banner, json.dumps({"lesion_coordinates": lesion_coordinates}, indent=2)

# Build the Gradio UI
with gr.Blocks(title="MedVision-Bench Playground", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# MedVision-Bench: Single-Image Playground")
    gr.Markdown("Upload an image and select a model to test the diagnostic pipeline and Human-in-the-Loop confidence routing.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Medical Image")
            model_dropdown = gr.Dropdown(
                choices=[
                    "gpt-4o-mini", 
                    "meta/llama-3.2-90b-vision-instruct", 
                    "qwen/qwen3.5-397b-a17b", 
                    "Multi-Agent: Hybrid (GPT Maker + Qwen Checker)", 
                    "Multi-Agent: Open-Source (Llama Maker + Qwen Checker)"
                ],
                value="gpt-4o-mini",
                label="Select Model"
            )
            submit_btn = gr.Button("Run Inference", variant="primary")
            
        with gr.Column():
            status_output = gr.Markdown("### Status: ⚪ Waiting for input...")
            
            with gr.Row():
                grade_output = gr.Textbox(label="Predicted Clinical Grade (0-4)")
                confidence_output = gr.Textbox(label="Confidence Score (%)")
                
            lesion_output = gr.Textbox(label="Raw Extraction: lesion_coordinates")

    # Wire up the logic
    submit_btn.click(
        fn=process_image,
        inputs=[image_input, model_dropdown],
        outputs=[grade_output, confidence_output, status_output, lesion_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
