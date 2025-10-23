import streamlit as st
import requests
import json
import re

# --- Ollama Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

# --- Streamlit Setup ---
st.set_page_config(page_title="AI Ticket Priority Classifier", page_icon="🎫", layout="centered")
st.title("🎯 AI-Powered Ticket Priority Classifier (Llama3.2 + Streamlit)")
st.markdown("""
This app uses your **local Llama 3.2 model (via Ollama)** to classify support tickets
into **High**, **Medium**, or **Low priority** using NLP reasoning.
""")

# --- User Inputs ---
subject = st.text_input("📨 Ticket Subject:")
body = st.text_area("📝 Ticket Body / Description:", height=200)

if st.button("🚀 Classify Priority"):
    if not (subject or body):
        st.warning("Please enter at least a subject or body before classifying.")
    else:
        ticket_text = f"Subject: {subject}\nBody: {body}"

        prompt = f"""
You are an NLP assistant trained to assess the urgency of customer support tickets.
Classify the ticket as High, Medium, or Low priority.
Respond with only one word: High, Medium, or Low.

Ticket:
{ticket_text}
        """

        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={"model": MODEL_NAME, "prompt": prompt},
                timeout=60
            )

            if response.status_code == 200:
                raw_text = response.text.strip()

                # --- FIX: Extract all JSON objects using regex ---
                json_objects = re.findall(r'\{.*?\}', raw_text)
                priority = None

                for obj_str in json_objects:
                    try:
                        obj = json.loads(obj_str)
                        resp = obj.get("response", "").strip()
                        if resp:  # take the first non-empty response
                            priority = resp.capitalize()
                            break
                    except:
                        continue

                if priority:
                    if priority.startswith("High"):
                        st.success("🟥 **Priority: HIGH** — Critical issue, needs immediate attention!")
                    elif priority.startswith("Medium"):
                        st.info("🟧 **Priority: MEDIUM** — Important but not urgent.")
                    elif priority.startswith("Low"):
                        st.write("🟩 **Priority: LOW** — Routine or minor query.")
                    else:
                        st.warning(f"🤔 Unexpected response from Ollama: {priority}")
                else:
                    st.error("⚠️ Could not find a valid response from Ollama.")

            else:
                st.error(f"❌ Ollama API returned {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f"⚠️ Connection error: {e}")

st.markdown("---")
st.caption("Built with 🧠 Streamlit + Ollama (Llama3.2) for NLP-based Ticket Classification.")
