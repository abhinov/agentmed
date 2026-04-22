import gradio as gr
import json
import random

def process_image(image, model):
    """
    Dummy inference function simulating the JSON schema response from the model.
    """
    if image is None:
        return "N/A", "N/A", "### Status: ⚪ Waiting for input...", "{}"
    
    # Simulate model prediction
    predicted_grade = random.choice([0, 1, 2, 3, 4])
    confidence_score = random.randint(60, 99)
    
    # Determine HITL Status banner
    if confidence_score >= 85:
        status_banner = "### Status: 🟢 Auto-Triage Approved"
    else:
        status_banner = "### Status: 🔴 Human-in-the-Loop Review Required"
        
    # Simulate raw JSON schema extraction for lesion coordinates
    dummy_lesions = {
        "lesion_coordinates": [
            {"type": "microaneurysm", "x_coord": 256, "y_coord": 512, "confidence": 0.94},
            {"type": "hard_exudate", "x_coord": 128, "y_coord": 800, "confidence": 0.88}
        ]
    }
    
    return str(predicted_grade), str(confidence_score), status_banner, dummy_lesions

# Build the Gradio UI
with gr.Blocks(title="MedVision-Bench Playground", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# MedVision-Bench: Single-Image Playground")
    gr.Markdown("Upload an image and select a model to test the diagnostic pipeline and Human-in-the-Loop confidence routing.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Medical Image")
            model_dropdown = gr.Dropdown(
                choices=["gpt-4o", "gpt-4o-mini", "gemini-1.5-pro", "gemini-1.5-flash"],
                value="gpt-4o",
                label="Select Model"
            )
            submit_btn = gr.Button("Run Inference", variant="primary")
            
        with gr.Column():
            status_output = gr.Markdown("### Status: ⚪ Waiting for input...")
            
            with gr.Row():
                grade_output = gr.Textbox(label="Predicted Clinical Grade (0-4)")
                confidence_output = gr.Textbox(label="Confidence Score (%)")
                
            lesion_output = gr.JSON(label="Raw Extraction: lesion_coordinates")

    # Wire up the logic
    submit_btn.click(
        fn=process_image,
        inputs=[image_input, model_dropdown],
        outputs=[grade_output, confidence_output, status_output, lesion_output]
    )

if __name__ == "__main__":
    demo.launch()
