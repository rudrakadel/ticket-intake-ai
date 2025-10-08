import streamlit as st
from transformers import pipeline
import spacy

# ------------------------
# Load AI models
# ------------------------
st.sidebar.subheader("Loading AI models... Please wait")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
nlp = spacy.load("en_core_web_sm")
st.sidebar.success("AI models loaded!")

# Candidate labels for semantic tagging
CANDIDATE_LABELS = ["Technical Support", "Billing", "Account Management", "General Inquiry", "Server Issue", "Network Issue"]

# ------------------------
# Streamlit Page
# ------------------------
st.title("🤖 AI Ticket Classifier & Semantic Tagger")

st.markdown("""
Enter a support ticket below, and the app will:
- Classify it into categories using Hugging Face Transformers
- Optionally extract named entities using spaCy
""")

# ------------------------
# Ticket Input
# ------------------------
ticket_text = st.text_area("Enter your ticket text here", height=200)

ner_checkbox = st.checkbox("Extract Named Entities (NER) using spaCy")

if st.button("Classify Ticket"):
    if not ticket_text.strip():
        st.warning("Please enter ticket text first")
    else:
        # ------------------------
        # Hugging Face Classification
        # ------------------------
        result = classifier(ticket_text, candidate_labels=CANDIDATE_LABELS)
        top_label = result['labels'][0]
        st.subheader("🟢 Predicted Ticket Category")
        st.write(f"**{top_label}** (Confidence: {result['scores'][0]:.2f})")

        st.subheader("All Scores")
        for label, score in zip(result['labels'], result['scores']):
            st.write(f"{label}: {score:.2f}")

        # ------------------------
        # spaCy Named Entity Recognition
        # ------------------------
        if ner_checkbox:
            doc = nlp(ticket_text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            if entities:
                st.subheader("🔹 Named Entities")
                for ent_text, ent_label in entities:
                    st.write(f"{ent_text} ({ent_label})")
            else:
                st.info("No named entities found.")
