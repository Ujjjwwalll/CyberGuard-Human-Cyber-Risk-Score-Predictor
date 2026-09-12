# ================================
# STEP 1: Import required libraries
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import altair as alt



# ================================
# STEP 2: Load all saved model/encoder/scaler files
# (These were created and saved in our Jupyter notebook)
# ================================
model = joblib.load("risk_model.pkl")
ord_enc = joblib.load("ordinal_encoder.pkl")
scaler = joblib.load("scaler.pkl")
ordinal_cols = joblib.load("ordinal_cols.pkl")
model_columns = joblib.load("model_columns.pkl")
risk_map = joblib.load("risk_map.pkl")
reverse_risk_map = joblib.load("reverse_risk_map.pkl")



st.set_page_config(page_title="CyberGuard - Risk Assessment", page_icon="🛡️", layout="centered")



# ================================
# STEP 3: Questions, grouped into 4 sections for a clearer flow
# IMPORTANT: option order must match the order used in ordinal_encoder training
# ================================
question_sections = {
    "Passwords & Authentication": {
        "password_reuse": {
            "text": "Do you reuse the same password across multiple accounts?",
            "options": ["Never", "Sometimes", "Often", "Always"]
        },
        "password_strength": {
            "text": "How would you describe your typical password?",
            "options": ["Simple (name/birthdate/common word)", "Medium (word + numbers)", "Long & random (12+ chars, mixed)"]
        },
        "password_change_freq": {
            "text": "How often do you change your passwords?",
            "options": ["Never", "Rarely", "Every 6-12 months", "Every 1-3 months"]
        },
        "mfa_usage": {
            "text": "Do you use Multi-Factor Authentication (MFA) on work accounts?",
            "options": ["Never", "Sometimes", "Always"]
        },
    },
    "Email & Social Engineering": {
        "click_unknown_links": {
            "text": "Would you click a link in an urgent-looking email from an unknown sender?",
            "options": ["Never", "Rarely", "Sometimes", "Yes, likely"]
        },
        "verify_sender": {
            "text": "Before opening an attachment, do you verify the sender's identity?",
            "options": ["Never", "Sometimes", "Always"]
        },
        "report_suspicious_email": {
            "text": "If you receive a suspicious email, what do you usually do?",
            "options": ["Report to IT/Security", "Delete it", "Ignore it", "Open it to check"]
        },
        "past_phishing_victim": {
            "text": "Have you ever clicked a phishing link or fallen for a scam before?",
            "options": ["Never", "Once", "More than once"]
        },
    },
    "Devices & Network": {
        "public_wifi_usage": {
            "text": "Do you access work accounts/emails on public Wi-Fi?",
            "options": ["Never", "Rarely", "Frequently"]
        },
        "unknown_usb_usage": {
            "text": "Would you plug in a USB drive from an unknown source?",
            "options": ["Never", "Only after scanning it", "Yes, if it looks harmless"]
        },
        "social_media_oversharing": {
            "text": "Do you share details about your job/workplace on social media?",
            "options": ["Never", "Occasionally", "Frequently"]
        },
        "screen_lock_habit": {
            "text": "Do you lock your screen when stepping away from your desk?",
            "options": ["Always", "Sometimes", "Never"]
        },
        "credential_sharing": {
            "text": "Have you ever shared your work password/login with a colleague?",
            "options": ["Never", "Once", "Occasionally"]
        },
    },
    "Awareness & Training": {
        "training_completion": {
            "text": "Have you completed your company's security awareness training?",
            "options": ["Yes, recently (within 6 months)", "Yes, but over a year ago", "Never"]
        },
        "training_engagement": {
            "text": "How seriously do you take security training sessions?",
            "options": ["Very seriously, apply what I learn", "Attend but forget", "Skip when possible"]
        },
    },
}



device_options = ["Company-issued & managed", "Personal device (BYOD)", "Shared/public device"]
department_options = ["Finance", "HR", "IT", "Sales & Marketing", "Operations", "Other"]

# Purely cosmetic: one small icon per section, used only for headers/labels below.
# Does not touch any question text, option order, or keys the model relies on.
section_icons = {
    "Passwords & Authentication": "🔑",
    "Email & Social Engineering": "📧",
    "Devices & Network": "💻",
    "Awareness & Training": "🎓",
}



# ================================
# STEP 2b: Custom styling
# Everything in this block is just visual - none of it touches the ML logic
# ================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');



html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3 {
    font-family: 'Sora', sans-serif !important;
    letter-spacing: -0.01em;
}



.stApp {
    background: radial-gradient(circle at 15% 0%, #10182c 0%, #0B1220 45%) fixed;
    color: #E6E9F0;
}



/* ---------- Hero header ---------- */
.hero-wrap {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    padding: 1.1rem 1.4rem;
    background: linear-gradient(135deg, rgba(79,209,197,0.14), rgba(79,209,197,0.02));
    border: 1px solid #1E2A44;
    border-radius: 14px;
    margin-bottom: 1.6rem;
}
.hero-icon {
    font-size: 2.1rem;
    line-height: 1;
    background: rgba(79,209,197,0.15);
    border-radius: 12px;
    padding: 0.5rem 0.65rem;
}
.hero-title {
    font-family: 'Sora', sans-serif;
    font-weight: 800;
    font-size: 1.7rem;
    color: #F3F6FB;
    margin: 0;
    line-height: 1.1;
}
.hero-subtitle {
    color: #8A93A6;
    font-size: 0.95rem;
    margin-top: 0.15rem;
}



/* ---------- Section cards ---------- */
.section-card {
    background: #121A2C;
    border: 1px solid #22304A;
    border-radius: 12px;
    padding: 1.1rem 1.3rem 0.4rem 1.3rem;
    margin-top: 1.3rem;
    margin-bottom: 0.4rem;
}
.section-label {
    font-family: 'Sora', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #4FD1C5;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-divider {
    border: none;
    border-top: 1px solid #26314A;
    margin: 0.6rem 0 1rem 0;
}
.field-wrap {
    padding-bottom: 0.6rem;
    margin-bottom: 0.6rem;
    border-bottom: 1px dashed #1E2A44;
}
.field-wrap:last-child {
    border-bottom: none;
}



/* progress pill above the form */
.progress-pill {
    display: inline-block;
    background: #141B2D;
    border: 1px solid #26314A;
    color: #8A93A6;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    font-size: 0.82rem;
    margin-bottom: 0.9rem;
}



/* primary button */
.stButton > button, .stFormSubmitButton > button {
    background-color: #4FD1C5;
    color: #0B1220;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 1.6rem;
    transition: transform 0.05s ease-in-out, background-color 0.15s ease-in-out;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    background-color: #6EE0D6;
    color: #0B1220;
}
.stButton > button:active, .stFormSubmitButton > button:active {
    transform: scale(0.98);
}



/* result status card */
.status-card {
    border-left: 5px solid;
    background: linear-gradient(135deg, #141B2D 0%, #111827 100%);
    padding: 1.1rem 1.3rem;
    border-radius: 10px;
    margin-top: 0.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
}
.status-title {
    font-family: 'Sora', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.status-desc {
    color: #A6AEC2;
    font-size: 0.95rem;
    line-height: 1.5;
}



/* chat panel */
[data-testid="stChatMessage"] {
    background-color: #141B2D;
    border: 1px solid #26314A;
    border-radius: 10px;
}
.chat-heading {
    font-family: 'Sora', sans-serif;
    color: #4FD1C5;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    margin-bottom: 0.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.chat-caption {
    color: #8A93A6;
    font-size: 0.88rem;
    margin-bottom: 0.7rem;
}



/* Fix 1: reduce the large default top margin above the page content */
.block-container {
    padding-top: 2.4rem;
    max-width: 760px;
}



/* Fix 2: bigger question text - multiple selectors as fallbacks since
   Streamlit's internal DOM structure/class names vary by version */
.stApp div[data-testid="stWidgetLabel"] p,
.stApp div[data-testid="stWidgetLabel"] label,
.stApp div[data-testid="stWidgetLabel"] {
    font-size: 1.18rem !important;
    line-height: 1.5 !important;
    color: #E9ECF3 !important;
}



/* Fix 2b: bigger option text (the actual MCQ choices, not just the question) */
.stApp [data-testid="stRadio"] label,
.stApp [data-testid="stRadio"] label p,
.stApp [data-testid="stRadio"] div[role="radiogroup"] label,
.stApp [data-testid="stRadio"] span {
    font-size: 1.05rem !important;
    line-height: 1.6 !important;
}



hr {
    border-color: #1E2A44 !important;
}



/* subtle scrollbar for the chat container */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-thumb { background: #26314A; border-radius: 4px; }
::-webkit-scrollbar-track { background: transparent; }
</style>
""", unsafe_allow_html=True)



# ================================
# STEP 2c: Sidebar - context + reset control
# ================================
with st.sidebar:
    st.markdown("### 🛡️ CyberGuard")
    st.caption("Human Cyber-Risk Score Predictor")
    with st.expander("How this works", expanded=False):
        st.write(
            "Answer 17 short questions about your day-to-day security habits. "
            "A trained ML model estimates your risk level, and a local AI assistant "
            "is available afterward to explain the result and suggest improvements."
        )
    st.markdown("---")
    st.markdown("##### Risk levels")
    st.markdown(
        "- 🟢 **Low** — strong habits\n"
        "- 🟡 **Moderate** — a few gaps\n"
        "- 🟠 **High** — meaningful exposure\n"
        "- 🔴 **Severe** — needs prompt attention"
    )
    st.markdown("---")
    if st.button("🔄 Restart Assessment", use_container_width=True):
        # bump the version counter -> every question widget below gets a brand new
        # key, so Streamlit mounts fresh unselected widgets instead of trying
       
        st.session_state.form_version = st.session_state.get("form_version", 0) + 1
        for key in ["prediction_made", "predicted_label", "user_answers_snapshot",
                    "chat_history", "prediction_proba"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()



st.markdown("""
<div class="hero-wrap">
    <div class="hero-icon">🛡️</div>
    <div>
        <p class="hero-title">CyberGuard</p>
        <p class="hero-subtitle">Cybersecurity risk assessment based on your everyday security habits</p>
    </div>
</div>
""", unsafe_allow_html=True)



# ================================
# STEP 6a: Function to talk to the local LLM (Ollama)
# ================================
def ask_local_llm(prompt):
    """
    Sends a prompt to the locally running Ollama model and returns its text response.
    Make sure Ollama is running in the background (ollama serve / ollama run llama3.2).
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:latest",
                "prompt": prompt,
                "stream": False
            }
        )
        response_data = response.json()
        return response_data.get("response", "Sorry, no response received from the model.")
    except Exception as e:
        return f"⚠️ Could not reach local LLM. Make sure Ollama is running. Error: {e}"



# ================================
# STEP 6b: Session state
# ================================
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
if "predicted_label" not in st.session_state:
    st.session_state.predicted_label = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



# ================================
# STEP 4: Build the form
# (question_sections / device_options / department_options are already defined above)
# ================================
# ================================
st.markdown("### 📝 Security Behavior Questionnaire")

total_questions = sum(len(qs) for qs in question_sections.values()) + 2  # +device_type +department
st.markdown(f"<span class='progress-pill'>{total_questions} quick questions · about 3 minutes</span>", unsafe_allow_html=True)



user_answers = {}
v = st.session_state.get("form_version", 0)  # bumped by Restart button to force fresh widgets



with st.form(f"risk_form_{v}"):
    for section_name, questions in question_sections.items():
        icon = section_icons.get(section_name, "")
        st.markdown(f"<div class='section-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-label'>{icon} {section_name}</div>", unsafe_allow_html=True)
        for col_name, q in questions.items():
            st.markdown("<div class='field-wrap'>", unsafe_allow_html=True)
            # the widget's own key is versioned (so Restart resets it), but we still
            # store the answer under the plain col_name - that's what the model expects
            user_answers[col_name] = st.radio(q["text"], q["options"], key=f"{col_name}_{v}", index=None)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)



    st.markdown(f"<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>🧑‍💼 About Your Role</div>", unsafe_allow_html=True)
    st.markdown("<div class='field-wrap'>", unsafe_allow_html=True)
    device_type = st.radio("What device do you primarily use for work?", device_options, key=f"device_type_{v}", index=None)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='field-wrap'>", unsafe_allow_html=True)
    department = st.radio("Which department do you work in?", department_options, key=f"department_{v}", index=None)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)



    submitted = st.form_submit_button("🚀 Check My Risk Score", use_container_width=True)



# ================================
# STEP 5: What happens when "Check My Risk Score" is pressed
# ================================
if submitted:
    input_data = user_answers.copy()
    input_data["device_type"] = device_type
    input_data["department"] = department



    # ---- 5a: Validate every question was actually answered ----
    unanswered = [k for k, v in input_data.items() if v is None]
    if unanswered:
        st.warning(f"Please answer all questions before checking your score. Missing: {len(unanswered)} question.")
    else:
        input_df = pd.DataFrame([input_data])



        with st.expander("Your submitted answers", expanded=False):
            st.dataframe(input_df, use_container_width=True)



        # ---- 5b: Encode ordinal columns using the SAME encoder from training ----
        input_df[ordinal_cols] = ord_enc.transform(input_df[ordinal_cols])



        # ---- 5c: Manually one-hot encode device_type and department ----
        final_input = pd.DataFrame(0, index=[0], columns=model_columns)
        for col in ordinal_cols:
            final_input[col] = input_df[col].values[0]



        device_col = f"device_type_{device_type}"
        department_col = f"department_{department}"
        if device_col in final_input.columns:
            final_input[device_col] = 1
        if department_col in final_input.columns:
            final_input[department_col] = 1



        final_input = final_input[model_columns]



        # ---- 5d: Scale, using the SAME scaler from training ----
        final_input_scaled = scaler.transform(final_input)



        # ---- 5e: Predict ----
        prediction = model.predict(final_input_scaled)[0]
        prediction_proba = model.predict_proba(final_input_scaled)[0]
        predicted_label = reverse_risk_map[prediction]



        # ---- 5f: Store the result in session_state instead of rendering it directly ----
        # (rendering happens in the persistent block below, so it survives future
        # reruns caused by chatting with the LLM, instead of disappearing)
        st.session_state.prediction_made = True
        st.session_state.predicted_label = predicted_label
        st.session_state.prediction_proba = prediction_proba.tolist()
        st.session_state.user_answers_snapshot = input_data



# ================================
# STEP 5f (persistent): Always show the result if we have one
# This runs on EVERY rerun (including ones triggered by the chat box),
# ================================
if st.session_state.prediction_made:
    st.write("---")
    st.markdown("### 📊 Your Risk Assessment Result")



    predicted_label = st.session_state.predicted_label
    prediction_proba = st.session_state.prediction_proba



    risk_colors = {"Low": "#34D399", "Moderate": "#FBBF24", "High": "#FB923C", "Severe": "#F87171"}
    risk_icons = {"Low": "🟢", "Moderate": "🟡", "High": "🟠", "Severe": "🔴"}
    risk_descriptions = {
        "Low": "Your habits reflect strong security awareness. Keep it up.",
        "Moderate": "A few habits could use attention before they become a real risk.",
        "High": "Several habits meaningfully increase your exposure to attacks.",
        "Severe": "Multiple high-risk habits combined - this needs prompt attention."
    }
    color = risk_colors[predicted_label]
    icon = risk_icons[predicted_label]



    st.markdown(f"""
    <div class="status-card" style="border-left-color:{color};">
        <div class="status-title" style="color:{color};">{icon} {predicted_label} Risk</div>
        <div class="status-desc">{risk_descriptions[predicted_label]}</div>
    </div>
    """, unsafe_allow_html=True)



    proba_df = pd.DataFrame({
        "Risk Level": [reverse_risk_map[i] for i in range(len(prediction_proba))],
        "Probability": prediction_proba
    })
    order = ["Low", "Moderate", "High", "Severe"]
    proba_df["Risk Level"] = pd.Categorical(proba_df["Risk Level"], categories=order, ordered=True)
    proba_df = proba_df.sort_values("Risk Level")



    chart = alt.Chart(proba_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("Risk Level:N", sort=order, axis=alt.Axis(labelAngle=0, labelFontSize=13, title=None)),
        y=alt.Y("Probability:Q", axis=alt.Axis(format="%")),
        color=alt.Color("Risk Level:N",
                         scale=alt.Scale(domain=order, range=[risk_colors[l] for l in order]),
                         legend=None),
        tooltip=[alt.Tooltip("Risk Level:N"), alt.Tooltip("Probability:Q", format=".1%")]
    ).properties(height=260).configure_view(strokeWidth=0)
    st.altair_chart(chart, use_container_width=True)



# ================================
# STEP 7: Chat box powered by local LLM
# ================================
if st.session_state.prediction_made:
    st.write("---")
    st.markdown("<div class='chat-heading'>🤖 Ask CyberGuard AI Assistant</div>", unsafe_allow_html=True)
    st.markdown("<div class='chat-caption'>Ask why you got this risk score, or how to improve it.</div>", unsafe_allow_html=True)





    chat_box = st.container(height=330, border=True)



    with chat_box:
        if not st.session_state.chat_history:
            st.caption("💬 Try asking: \"Why did I get this score?\" or \"What's my biggest risk area?\"")
        for role, message in st.session_state.chat_history:
            avatar = "🧑" if role == "user" else "🛡️"
            with st.chat_message(role, avatar=avatar):
                st.write(message)



    user_question = st.chat_input("Type your question here...")



    if user_question:
        st.session_state.chat_history.append(("user", user_question))



        with chat_box:
            with st.chat_message("user", avatar="🧑"):
                st.write(user_question)



            context_prompt = f"""
You are a friendly cybersecurity awareness assistant.
A user just completed a risk assessment quiz and got the result: {st.session_state.predicted_label} risk.



Their answers were: {st.session_state.user_answers_snapshot}



The user is now asking: "{user_question}"  



Give a short, clear, non-technical answer, and if relevant, give 1-2 practical tips.
"""
            with st.chat_message("assistant", avatar="🛡️"):
                with st.spinner("Thinking..."):
                    llm_response = ask_local_llm(context_prompt)
                    st.write(llm_response)



        st.session_state.chat_history.append(("assistant", llm_response))
