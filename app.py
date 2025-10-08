import streamlit as st
import os
import uuid

# Local folder to save uploads
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.set_page_config(page_title="AI Knowledge Engine", page_icon="🤖", layout="centered")

st.title("🤖 AI Powered Knowledge Engine")
st.subheader("Smart Ticket Upload System")

st.write("Upload a ticket file or paste text below:")

# Input options
uploaded_file = st.file_uploader("Choose a ticket file", type=["txt", "log", "md", "json"])
ticket_text = st.text_area("Or paste ticket text here")

if st.button("Upload Ticket"):
    if uploaded_file:
        # Save uploaded file locally
        ticket_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_FOLDER, f"{ticket_id}_{uploaded_file.name}")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.success(f"✅ Done and uploaded! (File saved as {uploaded_file.name})")
        st.info(f"Ticket ID: `{ticket_id}`")
    elif ticket_text.strip():
        # Save typed text as a new file
        ticket_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_FOLDER, f"{ticket_id}_text_ticket.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(ticket_text)
        st.success("✅ Done and uploaded! (Text saved as file)")
        st.info(f"Ticket ID: `{ticket_id}`")
    else:
        st.warning("⚠️ No file or text provided. Please upload or enter ticket details.")

# Optional clean styling
st.markdown("""
<style>
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.6em 1.5em;
    }
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #ccc;
    }
</style>
""", unsafe_allow_html=True)
