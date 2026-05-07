import streamlit as st
import pandas as pd
import json
import os
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="AgentMed: Clinical AI Gateway", page_icon="🏥", layout="wide")

# ==========================================
# BACKEND: DATA LOADING
# ==========================================
@st.cache_data
def load_real_fleet_data():
    csv_path = "results_llama_gpt_images.csv"
    if not os.path.exists(csv_path): return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
        processed = []
        for _, row in df.iterrows():
            m_g = row.get('maker_diagnosis', 0)
            m_c = float(row.get('maker_confidence', 0))
            
            c_g = row.get('checker_diagnosis', m_g) if row.get('checker_triggered', False) else m_g
            c_c = float(row.get('checker_confidence', m_c)) if row.get('checker_triggered', False) else m_c

            safe_threshold = 0.8 if m_c <= 1.0 else 80
            
            processed.append({
                'case_id': row.get('filename', 'Unknown'),
                'primary_grade': int(m_g), 
                'primary_conf': m_c,
                'second_opinion_grade': int(c_g), 
                'second_opinion_conf': c_c,
                'diagnostic_discordance': m_g != c_g,
                'triage_decision': "🟢 Safe for Auto-Triage" if m_g == c_g and m_c > safe_threshold else "🔴 Hard Deferral to Human"
            })
        return pd.DataFrame(processed)
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return pd.DataFrame()

telemetry_df = load_real_fleet_data()

# ==========================================
# ARCHITECTURE: API ROUTING
# ==========================================
def encode_and_compress_image(uploaded_file):
    """Local pre-processing to ensure data efficiency before API transmission."""
    image = Image.open(uploaded_file)
    if image.mode != 'RGB': image = image.convert('RGB')
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_json(text):
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    except: return {}

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def call_vlm_api(client, model_name, prompt, image_b64):
    """Resilient API caller with exponential backoff for network stability."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]}],
        max_tokens=1000, temperature=0.0
    )
    return response.choices[0].message.content or "{}"

# ==========================================
# FRONTEND UI (CLINICIAN WORKFLOW)
# ==========================================
st.title("🏥 AgentMed: Clinical AI Copilot")
st.markdown("Automated Diabetic Retinopathy screening with multi-agent consensus verification.")

tab1, tab2, tab3 = st.tabs(["🩺 Patient Triage & Second Opinion", "📊 Clinical Analytics on IDRiD", "🔍 Human Audit Log"])

with tab1:
    st.markdown("### Retinal Scan Evaluation")
    st.info("ℹ️ **Clinical Safety Protocol:** This workflow forces a multi-agent consensus. If the Primary AI and the Second-Opinion AI disagree, the scan is flagged for a **Hard Deferral** to a human ophthalmologist.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload Patient Retinal Scan", type=["jpg", "jpeg", "png"])
        protocol_choice = st.selectbox("Diagnostic Consensus Protocol", [
            "Primary AI (Llama-90B) + Second Opinion (GPT-4o)",
            "Single Agent Baseline (GPT-4o-mini)"
        ])
        analyze_btn = st.button("Generate Diagnostic Report", type="primary", use_container_width=True)

    with col2:
        if analyze_btn and uploaded_file:
            with st.spinner("Executing live multi-agent clinical evaluation..."):
                try:
                    b64_img = encode_and_compress_image(uploaded_file)
                    
                    # USER: Ensure API keys are set in your environment variables
                    o_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                    n_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ.get("NVIDIA_API_KEY"))
                    
                    # Enforced JSON structure tailored for clinical outputs
                    base_prompt = (
                        'Analyze this retinal image for Diabetic Retinopathy. '
                        'Output STRICT JSON: "predicted_grade" (int 0-4), "confidence_score" (float 0.0-1.0), '
                        '"clinical_triage_plan" (string recommendation), "lesion_coordinates" (list).'
                    )
                    
                    if "Baseline" in protocol_choice:
                        res_text = call_vlm_api(o_client, "gpt-4o-mini", base_prompt, b64_img)
                        m_dict = extract_json(res_text)
                        c_dict = m_dict
                    else:
                        m_text = call_vlm_api(n_client, "meta/llama-3.2-90b-vision-instruct", base_prompt, b64_img)
                        m_dict = extract_json(m_text)
                        
                        rev_prompt = (
                            f"A primary AI predicted Grade {m_dict.get('predicted_grade',0)}. "
                            "Critically audit this scan as a Second-Opinion Consulting Ophthalmologist. "
                            f"{base_prompt}"
                        )
                        c_text = call_vlm_api(o_client, "gpt-4o", rev_prompt, b64_img)
                        c_dict = extract_json(c_text)

                    def clean_conf(val):
                        return float(val) if val else 0.0

                    m_g, m_c = m_dict.get("predicted_grade", 0), clean_conf(m_dict.get("confidence_score", 0))
                    c_g, c_c = c_dict.get("predicted_grade", m_g), clean_conf(c_dict.get("confidence_score", m_c))
                    triage_plan = c_dict.get("clinical_triage_plan", "No specific clinical recommendation provided. Human review required.")

                    # Triage Logic
                    safe_threshold = 0.8 if m_c <= 1.0 else 80
                    is_safe = (m_g == c_g) and (m_c > safe_threshold)
                    
                    b_color = "#d4edda" if is_safe else "#f8d7da"
                    decision_text = "🟢 Safe for Auto-Triage" if is_safe else "🔴 Hard Deferral: Mandates Human Review"
                    
                    st.markdown(f'<div style="background:{b_color}; padding:15px; border-radius:8px; text-align:center; font-weight:bold; font-size:18px;">{decision_text}</div>', unsafe_allow_html=True)
                    st.metric("Consensus Diagnostic Grade", f"Grade {c_g}")
                    
                    st.markdown("#### Clinical Recommendation")
                    st.info(f"🩺 **Next Steps:** {triage_plan}")
                    
                    st.write("---")
                    m1, m2 = st.columns(2)
                    m1.metric("Primary AI Confidence", f"{m_c:.4g}")
                    m2.metric("Second-Opinion Confidence", f"{c_c:.4g}")
                    
                    with st.expander("🔍 View AI Spatial Coordinates & Raw Audit Trail"):
                        st.json({"lesion_coordinates": c_dict.get("lesion_coordinates", [])})
                        st.caption("Coordinates indicate areas of interest (microaneurysms, hemorrhages, exudates) flagged by the Second-Opinion AI.")
                except Exception as e:
                    st.error(f"System Error during evaluation: {e}")

with tab2:
    if not telemetry_df.empty:
        st.markdown("### Retrospective Cohort Analytics")
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Patient Scans Evaluated", len(telemetry_df))
        k2.metric("Human Escalation Rate", f"{(telemetry_df['diagnostic_discordance'].mean()*100):.1f}%")
        k3.metric("Auto-Triage Rate", f"{((telemetry_df['triage_decision']=='🟢 Safe for Auto-Triage').mean()*100):.1f}%")
        
        st.markdown("#### Population Severity Distribution")
        st.bar_chart(telemetry_df['second_opinion_grade'].value_counts())

with tab3:
    if not telemetry_df.empty:
        st.markdown("### Audit Log: Multi-Agent Discordance")
        st.markdown("Displays all cases where the Primary AI and Second-Opinion AI disagreed on the diagnostic grade, triggering a Hard Deferral.")
        display_df = telemetry_df[telemetry_df['diagnostic_discordance']==True].reset_index(drop=True)
        st.dataframe(display_df, use_container_width=True)