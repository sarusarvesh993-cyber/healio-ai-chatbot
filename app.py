import streamlit as st
import os
import sys
import time
import json
from datetime import datetime
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from healthcare_chatbot import HealthcareChatbot
from report_analyzer import ReportAnalyzer
from drug_interaction import DrugInteractionEngine
from symptom_triage import SymptomTriageEngine
from pdf_summary import DoctorSummaryGenerator

st.set_page_config(
    page_title="MediCare AI Pro - Advanced Clinical Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    
    .main .block-container { max-width: 1140px; padding-top: 1.2rem; padding-bottom: 2.5rem; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .pro-header {
        background: linear-gradient(135deg, #0077B6 0%, #0096C7 45%, #48CAE4 100%);
        padding: 1.6rem 2rem 1.4rem;
        border-radius: 20px;
        margin-bottom: 1.2rem;
        color: white;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 30px rgba(0, 119, 182, 0.22);
    }
    .pro-header h1 { color: white; font-size: 1.85rem; font-weight: 700; margin: 0; }
    .pro-header .tagline { color: rgba(255,255,255,0.92); font-size: 0.95rem; margin: 4px 0 0 64px; }
    .pro-header .badges { display: flex; gap: 8px; margin-top: 12px; padding-left: 64px; flex-wrap: wrap; }
    .pro-header .badge {
        background: rgba(255,255,255,0.22);
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        color: white;
        border: 1px solid rgba(255,255,255,0.2);
    }

    .pro-disclaimer {
        background: #FFF8E1;
        border: 1px solid #FFE082;
        border-left: 5px solid #FFA000;
        border-radius: 12px;
        padding: 0.85rem 1.2rem;
        margin-bottom: 1.2rem;
        font-size: 0.85rem;
        color: #5D4037;
    }

    .anxiety-box {
        background: #F0FDF4;
        border: 1.5px solid #86EFAC;
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        text-align: center;
    }

    .breathing-circle {
        width: 90px;
        height: 90px;
        background: #3B82F6;
        border-radius: 50%;
        margin: 10px auto;
        animation: breathe 8s infinite ease-in-out;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 0.8rem;
    }

    @keyframes breathe {
        0%, 100% { transform: scale(0.7); background: #60A5FA; }
        50% { transform: scale(1.15); background: #2563EB; }
    }

    .feature-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.2rem 1rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0, 119, 182, 0.05);
    }
    .feature-card h3 { margin: 0 0 4px; font-size: 0.9rem; color: #1565C0; font-weight: 600; }
    .feature-card p { font-size: 0.78rem; color: #78909C; margin: 0; line-height: 1.4; }

    .scanner-card {
        background: #F8FAFC;
        border: 1.5px dashed #94A3B8;
        border-radius: 18px;
        padding: 1.6rem;
        text-align: center;
        margin-bottom: 1.2rem;
    }

    .report-result-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 1.8rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        margin-top: 1.2rem;
    }

    [data-testid="stSidebar"] { background: linear-gradient(180deg, #F8FBFF 0%, #EDF4FF 100%); }
    .sidebar-status-online {
        background: #E8F5E9; border: 1px solid #A5D6A7;
        border-radius: 10px; padding: 8px 12px;
        font-size: 0.82rem; color: #2E7D32; margin-bottom: 10px;
    }
    .sidebar-status-offline {
        background: #FFF3E0; border: 1px solid #FFE082;
        border-radius: 10px; padding: 8px 12px;
        font-size: 0.82rem; color: #E65100; margin-bottom: 10px;
    }
    .sidebar-stat {
        background: white; border: 1px solid #E3F2FD;
        border-radius: 10px; padding: 8px 12px;
        margin-bottom: 6px; font-size: 0.82rem; color: #37474F;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        margin-bottom: 1.2rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px 12px 0px 0px;
        padding: 10px 18px;
        font-weight: 600;
        font-size: 0.92rem;
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_session():
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.getenv("OPENROUTER_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    if "provider" not in st.session_state:
        st.session_state.provider = "auto"
    if "model_name" not in st.session_state:
        st.session_state.model_name = "meta-llama/llama-3.3-70b-instruct"
    if "language" not in st.session_state:
        st.session_state.language = "English"

    if "chatbot" not in st.session_state:
        st.session_state.chatbot = HealthcareChatbot(
            api_key=st.session_state.api_key,
            provider=st.session_state.provider,
            model_name=st.session_state.model_name
        )

    if "analyzer" not in st.session_state:
        st.session_state.analyzer = ReportAnalyzer(
            api_key=st.session_state.api_key,
            provider=st.session_state.provider,
            model_name=st.session_state.model_name
        )

    if "drug_engine" not in st.session_state:
        st.session_state.drug_engine = DrugInteractionEngine()

    if "triage_engine" not in st.session_state:
        st.session_state.triage_engine = SymptomTriageEngine()

    if "pdf_gen" not in st.session_state:
        st.session_state.pdf_gen = DoctorSummaryGenerator()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_report_analysis" not in st.session_state:
        st.session_state.last_report_analysis = None
    if "last_drug_analysis" not in st.session_state:
        st.session_state.last_drug_analysis = None
    if "last_triage_analysis" not in st.session_state:
        st.session_state.last_triage_analysis = None


def render_sidebar():
    with st.sidebar:
        st.markdown("### 🔑 Universal API Settings")

        provider_options = ["auto", "openrouter", "openai", "groq", "gemini"]
        provider_labels = ["⚡ Auto-Detect Provider", "🌐 OpenRouter (Any Model)", "🤖 OpenAI Direct (GPT-4o)", "⚡ Groq (Ultra-Fast)", "🔮 Google Gemini Direct"]
        
        selected_prov_idx = provider_options.index(st.session_state.get("provider", "auto")) if st.session_state.get("provider") in provider_options else 0
        chosen_provider = st.selectbox("API Provider", options=provider_options, format_func=lambda x: provider_labels[provider_options.index(x)], index=selected_prov_idx)

        current_key = st.session_state.get("api_key", "")
        new_api_key = st.text_input(
            "API Key (OpenRouter / OpenAI / Groq / Gemini)",
            value=current_key,
            type="password",
            placeholder="sk-or-... / sk-... / gsk_... / AIza...",
            help="Paste your API key here from any provider."
        )

        model_options = [
            "meta-llama/llama-3.3-70b-instruct",
            "openai/gpt-4o-mini",
            "gpt-4o-mini",
            "google/gemini-flash-1.5",
            "llama-3.3-70b-versatile",
            "deepseek/deepseek-chat",
            "anthropic/claude-3.5-haiku",
            "openrouter/auto",
            "Custom Model"
        ]

        current_model = st.session_state.get("model_name", "meta-llama/llama-3.3-70b-instruct")
        if current_model.endswith(":free"):
            current_model = current_model.replace(":free", "")

        selected_index = model_options.index(current_model) if current_model in model_options else 0
        selected_model = st.selectbox("LLM / Vision Model", options=model_options, index=selected_index)

        if selected_model == "Custom Model":
            custom_model = st.text_input("Custom Model ID", value=current_model if current_model not in model_options else "")
            chosen_model = custom_model.strip() if custom_model.strip() else current_model
        else:
            chosen_model = selected_model

        col1, col2 = st.columns(2)
        with col1:
            save_clicked = st.button("💾 Apply Key", use_container_width=True)
        with col2:
            st.markdown(
                '<a href="https://openrouter.ai/keys" target="_blank" style="display:inline-block;width:100%;text-align:center;padding:7px;border-radius:24px;background:#E3F2FD;color:#0077B6;text-decoration:none;font-size:0.82rem;font-weight:600;">Get Key</a>',
                unsafe_allow_html=True
            )

        if save_clicked or (new_api_key.strip() != current_key.strip() or chosen_model != current_model or chosen_provider != st.session_state.get("provider")):
            st.session_state.api_key = new_api_key.strip()
            st.session_state.provider = chosen_provider
            st.session_state.model_name = chosen_model.strip()
            st.session_state.chatbot.set_config(new_api_key.strip(), chosen_provider, chosen_model.strip())
            st.session_state.analyzer.update_config(new_api_key.strip(), chosen_provider, chosen_model.strip())
            st.rerun()

        if st.session_state.get("api_key"):
            st.markdown('<div class="sidebar-status-online">🟢 <strong>Status:</strong> Active & Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="sidebar-status-offline">⚠️ <strong>Status:</strong> API Key Required</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🌐 Language / भाषा / భాష")
        languages = ["English", "Hindi (हिंदी)", "Telugu (తెలుగు)", "Tamil (தமிழ்)", "Spanish (Español)", "French (Français)", "Arabic (العربية)", "German (Deutsch)"]
        selected_lang = st.selectbox("Response Language", options=languages, index=languages.index(st.session_state.get("language", "English")) if st.session_state.get("language") in languages else 0)
        if selected_lang != st.session_state.get("language"):
            st.session_state.language = selected_lang
            st.rerun()

        st.markdown("---")
        st.markdown("### 📊 Conversation Stats")
        summary = st.session_state.chatbot.get_conversation_summary()
        st.markdown(f'<div class="sidebar-stat"><strong>{summary["total_messages"]}</strong> total messages</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-stat"><strong>{summary["user_messages"]}</strong> questions asked</div>', unsafe_allow_html=True)

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chatbot.clear_conversation()
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.markdown('<div style="font-size:0.75rem;color:#64748B;text-align:center;">🏥 MediCare AI Pro • Multi-Modal Clinical Intelligence</div>', unsafe_allow_html=True)


def render_chat():
    chatbot = st.session_state.chatbot

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"#### 💬 AI Clinical Companion ({st.session_state.get('language', 'English')})")
    with col2:
        if st.button("🎙️ Voice Dictation Guide", help="Click for voice input guide"):
            st.info("💡 **Voice Input:** On Windows, press `Windows Key + H` while focused on the chat input to speak naturally in your language!")

    if not st.session_state.messages:
        st.markdown("""
        <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:12px;margin-bottom:1.2rem;">
            <div class="feature-card"><span style="font-size:1.8rem;display:block;margin-bottom:4px;">🩺</span><h3>Symptom Analysis</h3><p>Evidence-based triage & red flag checks</p></div>
            <div class="feature-card"><span style="font-size:1.8rem;display:block;margin-bottom:4px;">🥗</span><h3>Diet & Nutrition</h3><p>Heart-healthy and glycemic support</p></div>
            <div class="feature-card"><span style="font-size:1.8rem;display:block;margin-bottom:4px;">🩹</span><h3>First Aid Steps</h3><p>Emergency comfort & burn protocols</p></div>
        </div>
        """, unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Type or dictate your health question...", key="main_chat_input")

    if "user_input" in st.session_state and st.session_state.user_input:
        user_input = st.session_state.user_input
        st.session_state.user_input = None

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            active_key = st.session_state.get("api_key", "").strip()
            if not active_key:
                response = "⚠️ Please enter your API key in the sidebar and click 'Apply Key' to begin chatting."
                st.error(response)
            else:
                with st.spinner("Formulating evidence-based response..."):
                    result = chatbot.chat(
                        user_message=user_input,
                        api_key=active_key,
                        provider=st.session_state.get("provider", "auto"),
                        model_name=st.session_state.get("model_name", "meta-llama/llama-3.3-70b-instruct"),
                        language=st.session_state.get("language", "English")
                    )
                    response = result["response"]

                    if result.get("anxiety_detected"):
                        st.markdown("""
                        <div class="anxiety-box">
                            <h4 style="color:#166534;margin:0 0 6px 0;">🧘 High Distress Detected — Let's Calm Your Nervous System</h4>
                            <p style="color:#15803D;font-size:0.85rem;margin:0;">Breathe along with the circle: Inhale (4s), Hold (7s), Exhale (8s)</p>
                            <div class="breathing-circle">Breathe</div>
                        </div>
                        """, unsafe_allow_html=True)

                    message_placeholder = st.empty()
                    full_response = ""
                    for chunk in response.split(" "):
                        full_response += chunk + " "
                        time.sleep(0.01)
                        message_placeholder.markdown(full_response + "")
                    message_placeholder.markdown(full_response)

                    if result.get("sources"):
                        with st.expander("📚 Medical Knowledge Sources & References", expanded=False):
                            for src in result["sources"]:
                                st.markdown(f"- **{src['topic']}** — _{src['source']}_ ({src['category']})")

        st.session_state.messages.append({"role": "assistant", "content": response})


def render_report_scanner():
    st.markdown("### 📄 Medical Report & Prescription Scanner (Multi-Modal Vision)")
    st.markdown("Upload a photo or PDF of a **Blood Test, Lipid Profile, Thyroid Panel, CBC, or Doctor Prescription** for automated clinical parameter extraction.")

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Report or Prescription (JPG, PNG, WEBP, or PDF)",
            type=["png", "jpg", "jpeg", "webp", "pdf"],
            help="Ensure numbers and drug names are clearly readable."
        )

        custom_notes = st.text_area(
            "Optional: Add Patient Context or Symptoms",
            placeholder="e.g., Fasting for 12 hours, feeling mild fatigue and thirst...",
            height=80
        )

        analyze_button = st.button("🔍 Scan & Analyze Document", type="primary", use_container_width=True)

    with col2:
        if uploaded_file is not None:
            if "pdf" in uploaded_file.type.lower():
                st.info(f"📄 **PDF Document Loaded:** `{uploaded_file.name}` ({round(uploaded_file.size / 1024, 1)} KB)")
            else:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Document Preview", use_container_width=True)
        else:
            st.markdown("""
            <div class="scanner-card">
                <div style="font-size:2.4rem;margin-bottom:6px;">📸</div>
                <div style="font-weight:600;color:#1E293B;margin-bottom:4px;">Drag & Drop Lab Report / Prescription</div>
                <div style="font-size:0.85rem;color:#64748B;">Supports Blood Tests, Urine Panels, Lipid Profiles, Liver/Kidney Tests, & Prescriptions</div>
            </div>
            """, unsafe_allow_html=True)

    if analyze_button and uploaded_file is not None:
        active_key = st.session_state.get("api_key", "").strip()
        if not active_key:
            st.error("⚠️ Please enter an API Key in the sidebar to run the Vision Analyzer.")
        else:
            with st.spinner("🤖 Scanning document & extracting clinical parameters..."):
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                result = st.session_state.analyzer.analyze(
                    file_data=file_bytes,
                    file_type=uploaded_file.type,
                    custom_notes=custom_notes,
                    api_key=active_key,
                    provider=st.session_state.get("provider", "auto"),
                    model_name=st.session_state.get("model_name", "openai/gpt-4o-mini"),
                    language=st.session_state.get("language", "English")
                )

                if result["success"]:
                    st.session_state.last_report_analysis = result["analysis"]
                else:
                    st.error(result["error"])

    if st.session_state.last_report_analysis:
        st.markdown('<div class="report-result-box">', unsafe_allow_html=True)
        st.markdown(st.session_state.last_report_analysis)
        st.markdown('</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("💬 Discuss this Report in Chatbot", use_container_width=True):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Here is the report analysis we just completed:\n\n{st.session_state.last_report_analysis}\n\nWhat questions do you have regarding these lab findings or medications?"
                })
                st.info("Report attached to chat session! Switch to the 'AI Health Companion' tab to ask questions.")
        with col_b:
            st.download_button(
                label="📥 Download Clinical Analysis (.md)",
                data=st.session_state.last_report_analysis,
                file_name="medicare_report_analysis.md",
                mime="text/markdown",
                use_container_width=True
            )


def render_drug_interaction():
    st.markdown("### 💊 Drug Interaction & Medication Safety Engine")
    st.markdown("Check **contraindications, food-drug interactions, and administration schedules** using real-time OpenFDA data & clinical pharmacology AI.")

    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        drugs_input = st.text_input(
            "Enter Medications (separated by comma)",
            placeholder="e.g., Aspirin, Ibuprofen, Metformin, Atorvastatin",
            help="Type 1 or more brand/generic medication names."
        )

        patient_conditions = st.text_input(
            "Patient Health Conditions (Optional)",
            placeholder="e.g., Hypertension, Type 2 Diabetes, Peptic Ulcers, Asthma"
        )

        check_btn = st.button("🔬 Check Drug Interactions & Food Warnings", type="primary", use_container_width=True)

    with col2:
        st.info("💡 **Clinical Safety Check:** Enter combinations like `Aspirin, Ibuprofen` or `Lisinopril, Potassium` to check for synergistic toxicity or food restrictions.")

    if check_btn and drugs_input.strip():
        active_key = st.session_state.get("api_key", "").strip()
        if not active_key:
            st.error("⚠️ Please enter an API Key in the sidebar to run the Drug Safety Engine.")
        else:
            with st.spinner("Connecting to OpenFDA database & evaluating pharmacology matrix..."):
                drugs_list = [d.strip() for d in drugs_input.split(",") if d.strip()]
                result = st.session_state.drug_engine.check_interactions(
                    drugs=drugs_list,
                    patient_conditions=patient_conditions,
                    api_key=active_key,
                    provider=st.session_state.get("provider", "auto"),
                    model=st.session_state.get("model_name", "meta-llama/llama-3.3-70b-instruct"),
                    language=st.session_state.get("language", "English")
                )
                st.session_state.last_drug_analysis = result

    if st.session_state.last_drug_analysis:
        st.markdown('<div class="report-result-box">', unsafe_allow_html=True)
        st.markdown(st.session_state.last_drug_analysis)
        st.markdown('</div>', unsafe_allow_html=True)


def render_symptom_triage():
    st.markdown("### 🩺 Multi-Step Guided Symptom Checker & ESI Triage")
    st.markdown("Evaluate symptoms with clinical **Emergency Severity Index (ESI)** triage scoring to determine whether you need Emergency Care, Urgent Care, or Home Monitoring.")

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        body_part = st.selectbox(
            "1. Anatomical Region / Location",
            options=["Head & Neurological", "Chest & Cardiovascular", "Abdomen & Digestive", "Musculoskeletal & Joints", "Skin & Allergies", "Throat & Respiratory", "General / Systemic"]
        )

        primary_symptom = st.text_input(
            "2. Primary Symptom / Complaint",
            placeholder="e.g., Sharp throbbing headache with light sensitivity"
        )

        pain_level = st.slider("3. Pain Severity Scale (1 = Mild, 10 = Worst Possible Pain)", min_value=1, max_value=10, value=5)

        duration = st.selectbox(
            "4. Onset & Duration",
            options=["Sudden onset (< 2 hours)", "Developing over 2-12 hours", "1 to 3 days", "1 to 2 weeks", "Chronic (> 1 month)"]
        )

        age_group = st.selectbox(
            "5. Patient Age Group",
            options=["Adult (18-64)", "Elderly (65+)", "Adolescent (12-17)", "Child (2-11)", "Infant (< 2 years)"]
        )

    with col2:
        st.markdown("##### 6. Check Any Accompanying Symptoms:")
        symptom_options = [
            "Fever / Chills",
            "Nausea or Vomiting",
            "Dizziness / Lightheadedness",
            "Shortness of breath",
            "Chest pain radiating to arm/jaw",
            "Sudden weakness / numbness on one side",
            "Confusion / Slurred speech",
            "Stiff neck with headache"
        ]
        selected_associated = []
        for sym in symptom_options:
            if st.checkbox(sym, key=f"triage_sym_{sym}"):
                selected_associated.append(sym)

        triage_btn = st.button("🚨 Run Clinical ESI Triage Assessment", type="primary", use_container_width=True)

    if triage_btn and primary_symptom.strip():
        active_key = st.session_state.get("api_key", "").strip()
        if not active_key:
            st.error("⚠️ Please enter an API Key in the sidebar to run the Triage Engine.")
        else:
            with st.spinner("Computing ESI Acuity Score & evaluating emergency red flags..."):
                result = st.session_state.triage_engine.evaluate_triage(
                    body_part=body_part,
                    primary_symptom=primary_symptom,
                    pain_level=pain_level,
                    duration=duration,
                    associated_symptoms=selected_associated,
                    patient_age_group=age_group,
                    api_key=active_key,
                    provider=st.session_state.get("provider", "auto"),
                    model=st.session_state.get("model_name", "meta-llama/llama-3.3-70b-instruct"),
                    language=st.session_state.get("language", "English")
                )
                st.session_state.last_triage_analysis = result

    if st.session_state.last_triage_analysis:
        st.markdown('<div class="report-result-box">', unsafe_allow_html=True)
        st.markdown(st.session_state.last_triage_analysis)
        st.markdown('</div>', unsafe_allow_html=True)


def render_doctor_summary():
    st.markdown("### 📋 One-Click Doctor Visit Summary (Downloadable Clinical PDF)")
    st.markdown("Generate a structured clinical summary ready to print or email to your doctor before your consultation.")

    col1, col2 = st.columns([1, 1])

    with col1:
        patient_name = st.text_input("Patient Name", value="John Doe")
        age_gender = st.text_input("Age / Gender", value="38 / Male")
        chief_complaint = st.text_input("Chief Complaint", placeholder="e.g. Recurrent morning headaches and elevated fasting blood glucose")
        symptoms_history = st.text_area("Symptom Timeline & Severity", placeholder="e.g. Started 3 weeks ago, pain 6/10, worsening after meals", height=70)

    with col2:
        medications = st.text_input("Current Medications", placeholder="e.g. Metformin 500mg, Lisinopril 10mg")
        allergies = st.text_input("Known Allergies", value="Penicillin (mild rash)")
        vitals = st.text_input("Recorded Vitals", placeholder="e.g. BP: 130/85 mmHg, HR: 74 bpm, Fasting Glucose: 112 mg/dL")
        questions = st.text_area("Specific Questions to Ask Doctor", placeholder="e.g. 1. Should I undergo an HbA1c test?\n2. Is Metformin causing my morning nausea?", height=70)

    gen_btn = st.button("📄 Generate Printable Clinical Summary & PDF", type="primary", use_container_width=True)

    if gen_btn:
        chat_summary = ""
        if st.session_state.messages:
            chat_summary = "Recent AI consultation topics: " + "; ".join([m["content"][:80] for m in st.session_state.messages if m["role"] == "user"])

        md_content = st.session_state.pdf_gen.generate_markdown(
            patient_name=patient_name,
            age_gender=age_gender,
            chief_complaint=chief_complaint,
            symptoms_history=symptoms_history,
            medications=medications,
            allergies=allergies,
            vitals=vitals,
            questions=questions,
            conversation_summary=chat_summary
        )

        pdf_bytes = st.session_state.pdf_gen.generate_pdf_bytes(
            patient_name=patient_name,
            age_gender=age_gender,
            chief_complaint=chief_complaint,
            symptoms_history=symptoms_history,
            medications=medications,
            allergies=allergies,
            vitals=vitals,
            questions=questions,
            conversation_summary=chat_summary
        )

        st.markdown('<div class="report-result-box">', unsafe_allow_html=True)
        st.markdown(md_content)
        st.markdown('</div>', unsafe_allow_html=True)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                label="📥 Download Clinical PDF Report",
                data=pdf_bytes,
                file_name=f"Doctor_Summary_{patient_name.replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        with col_d2:
            st.download_button(
                label="📥 Download Markdown Summary (.md)",
                data=md_content,
                file_name=f"Doctor_Summary_{patient_name.replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True
            )


def main():
    load_css()
    initialize_session()
    render_sidebar()

    st.markdown("""
    <div class="pro-header">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
            <div style="font-size:2.2rem;background:rgba(255,255,255,0.22);width:52px;height:52px;display:flex;align-items:center;justify-content:center;border-radius:14px;">🏥</div>
            <h1 style="color:white;font-size:1.85rem;font-weight:700;margin:0;">MediCare AI Pro</h1>
        </div>
        <p class="tagline">Comprehensive Multi-Modal Clinical Intelligence & Diagnostic Companion</p>
        <div class="badges">
            <span class="badge">🔬 Multi-Modal Vision</span>
            <span class="badge">💊 OpenFDA Drug Engine</span>
            <span class="badge">🩺 ESI Clinical Triage</span>
            <span class="badge">📋 1-Click Doctor PDF</span>
            <span class="badge">🌐 Multi-Language</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="pro-disclaimer">
        <strong>Medical Disclaimer:</strong> MediCare AI Pro provides evidence-based health guidance and report analysis for educational and clinical consultation preparation. It is <strong>not</strong> a substitute for in-person emergency diagnosis or prescription. <strong>In an emergency, call 108 / 911 immediately.</strong>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 AI Health Companion",
        "📄 Report & Prescription Scanner",
        "💊 Drug Interactions",
        "🩺 Symptom Checker & Triage",
        "📋 Doctor Summary (PDF)"
    ])

    with tab1:
        render_chat()

    with tab2:
        render_report_scanner()

    with tab3:
        render_drug_interaction()

    with tab4:
        render_symptom_triage()

    with tab5:
        render_doctor_summary()


if __name__ == "__main__":
    main()
