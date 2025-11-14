# app.py
"""
AI Support Assistant Pro (Main App)

Features:
- Streamlit + FAISS RAG with Ollama model
- Short ticket IDs, requester email
- Structured knowledge_gap (flag, reason, kb_fill_text, filled_by, filled_at)
- Slack notifications (Incoming Webhook) with deep-links (Open/Close/Fill Gap)
- Reuse gap fills (KB_Gap_Fill) in RAG context for future tickets
- Analytics & History tabs
"""

import os
import io
import json
import time
import subprocess
import re
import uuid
import secrets
import base64
from typing import List, Dict, Any, Tuple
from datetime import datetime

import streamlit as st
st.set_page_config(page_title="AI Support Assistant Pro", layout="wide", page_icon="🧠")

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests  # Slack webhooks

# Optional dependencies
SENTIMENT_AVAILABLE = False
LANGDETECT_AVAILABLE = False
try:
    from textblob import TextBlob
    SENTIMENT_AVAILABLE = True
except ImportError:
    pass
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    pass

# ---------------- CONFIG ----------------
EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "llama3.2:latest"
FAISS_INDEX_FILE = "ticket_index.faiss"
METADATA_FILE = "ticket_metadata.json"
HISTORY_FILE = "conversation_history.json"

EMBED_DIM = 384
BATCH_SIZE = 512
APP_NAME = "AI Support Assistant Pro"

SLACK_WEBHOOK_URL = os.getenv(
    "SLACK_WEBHOOK_URL",
    "<Yourwebhook>"
)
ADMIN_APP_URL = os.getenv("ADMIN_APP_URL", "http://localhost:8502")  # admin deep-links

if not SENTIMENT_AVAILABLE:
    st.warning("⚠️ textblob not installed. Sentiment analysis disabled. Install: pip install textblob")
if not LANGDETECT_AVAILABLE:
    st.warning("⚠️ langdetect not installed. Language detection disabled. Install: pip install langdetect")

# ---------------- UTILITIES ----------------
def short_id(n: int = 9) -> str:
    # URL-safe short ID
    return base64.urlsafe_b64encode(secrets.token_bytes(n)).decode("utf-8").rstrip("=").lower()[:n]

def get_available_ollama_models() -> List[str]:
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")[1:]
            models = [line.split()[0] for line in lines if line.strip()]
            return models if models else [DEFAULT_LLM_MODEL]
    except Exception:
        pass
    return [DEFAULT_LLM_MODEL]

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_data
def batch_encode_cached(texts: List[str], model_name: str) -> np.ndarray:
    embedder = load_embedder()
    return embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)

def detect_sentiment(text: str) -> Tuple[str, float]:
    if not SENTIMENT_AVAILABLE:
        return "Neutral", 0.0
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1: return "Positive", polarity
        if polarity < -0.1: return "Negative", polarity
        return "Neutral", polarity
    except Exception:
        return "Neutral", 0.0

def detect_language(text: str) -> str:
    if not LANGDETECT_AVAILABLE or not text.strip():
        return "en"
    try:
        return detect(text)
    except Exception:
        return "en"

def extract_topics(text: str) -> List[str]:
    topics = []
    patterns = {
        "Login Issue": r"\b(login|sign in|password|authenticate|access)\b",
        "Payment": r"\b(payment|billing|invoice|charge|refund)\b",
        "Technical": r"\b(bug|error|crash|issue|problem|not working)\b",
        "Account": r"\b(account|profile|settings|delete|close)\b",
        "Feature Request": r"\b(feature|request|suggest|add|improve)\b",
        "Refund": r"\b(refund|money back|cancel|return)\b",
    }
    text_lower = text.lower()
    for topic, pattern in patterns.items():
        if re.search(pattern, text_lower):
            topics.append(topic)
    return topics if topics else ["General"]

def adjust_priority_by_sentiment(base_priority: str, sentiment: str, polarity: float) -> str:
    if sentiment == "Negative" and polarity < -0.3:
        return {"Low": "Medium", "Medium": "High", "High": "High"}.get(base_priority, base_priority)
    return base_priority

def save_metadata(metadata: List[Dict[str, Any]]):
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_metadata() -> List[Dict[str, Any]]:
    if not os.path.exists(METADATA_FILE):
        return []
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def build_index_from_dataframe(df: pd.DataFrame, embedder: SentenceTransformer):
    progress_bar = st.progress(0)
    st.info("🔄 Embedding dataset... please wait")

    required_cols = ["subject","body","answer","type","queue","priority","language","version"]
    for c in required_cols:
        if c not in df.columns:
            df[c] = ""

    texts, metadata = [], []
    total = len(df)
    for idx, row in df.iterrows():
        tags = [row.get(f"tag_{i}","") for i in range(1,9) if pd.notna(row.get(f"tag_{i}"))]
        text = (
            f"Subject: {row['subject']}\nBody: {row['body']}\n"
            f"Answer: {row['answer']}\nType: {row['type']}\n"
            f"Queue: {row['queue']}\nPriority: {row['priority']}\n"
            f"Language: {row['language']}\nTags: {', '.join(tags)}"
        )
        texts.append(text)
        metadata.append({
            "subject": str(row.get("subject","")),
            "body": str(row.get("body","")),
            "answer": str(row.get("answer","")),
            "type": str(row.get("type","")),
            "queue": str(row.get("queue","")),
            "priority": str(row.get("priority","")),
            "language": str(row.get("language","")),
            "version": str(row.get("version","")),
            "tags": tags
        })
        progress_bar.progress((idx + 1) / total)

    n = len(texts)
    embeddings = np.zeros((n, EMBED_DIM), dtype="float32")
    for i in range(0, n, BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        emb = embedder.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings[i:i+len(emb)] = emb.astype("float32")

    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_FILE)
    save_metadata(metadata)
    progress_bar.empty()
    st.success(f"✅ Indexed {n} tickets successfully!")

def load_index():
    if not os.path.exists(FAISS_INDEX_FILE):
        return None, []
    index = faiss.read_index(FAISS_INDEX_FILE)
    metadata = load_metadata()
    return index, metadata

def query_faiss(index, metadata, query_text: str, top_k: int = 3, filters: Dict = None):
    embedder = load_embedder()
    emb = embedder.encode([query_text], convert_to_numpy=True).astype("float32")
    search_k = top_k * 3 if filters else top_k
    D, I = index.search(emb, min(search_k, index.ntotal))

    results = []
    for idx in I[0]:
        if idx >= len(metadata):
            continue
        item = metadata[idx]
        if filters:
            if filters.get("queue") and item.get("queue") != filters["queue"]:
                continue
            if filters.get("priority") and item.get("priority") != filters["priority"]:
                continue
            if filters.get("language") and item.get("language") != filters["language"]:
                continue
        results.append(item)
        if len(results) >= top_k:
            break
    return results

def run_ollama_cli(model: str, prompt: str, timeout: int = 90) -> str:
    try:
        proc = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return "[ERROR] Ollama request timed out."
    except FileNotFoundError:
        return "[ERROR] Ollama not found. Please install Ollama: https://ollama.ai"
    except Exception as e:
        return f"[ERROR] Ollama CLI failed: {str(e)}"

    if proc.returncode != 0:
        return f"[ERROR] Ollama error: {proc.stderr.strip() or proc.stdout}"

    stdout = proc.stdout.strip()
    try:
        obj = json.loads(stdout)
        if "response" in obj:
            return obj["response"]
        if "message" in obj:
            return obj["message"].get("content", stdout)
    except Exception:
        pass

    json_objs = re.findall(r"\{[^{}]*\}", stdout, re.DOTALL)
    for s in json_objs:
        try:
            parsed = json.loads(s)
            if "priority" in parsed or "reply" in parsed:
                return s
        except Exception:
            continue
    return stdout

def parse_llm_response(raw_text: str) -> Tuple[str, str, str]:
    priority, reply, category = None, None, None
    try:
        data = json.loads(raw_text)
        priority = data.get("priority")
        reply = data.get("reply")
        category = data.get("category", "General")
        return priority, reply, category
    except Exception:
        pass
    lines = raw_text.split("\n")
    for line in lines:
        line_lower = line.lower()
        if "priority" in line_lower and not priority:
            match = re.search(r'(high|medium|low)', line, re.IGNORECASE)
            if match:
                priority = match.group(1).capitalize()
        if not reply:
            m = re.search(r'^\s*(reply|response|answer)\s*:\s*(.+)$', line, re.IGNORECASE)
            if m:
                reply = m.group(2).strip()
        if "category" in line_lower and not category:
            category = line.split(":", 1)[-1].strip()
    return priority, reply, category or "General"

def post_to_slack(text: str, blocks: list | None = None):
    url = SLACK_WEBHOOK_URL
    if not url:
        return
    payload = {"text": text}
    if blocks:
        payload["blocks"] = blocks
    try:
        resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        st.warning(f"Slack notification failed: {e}")

# ---------------- MAIN APP ----------------
embedder = load_embedder()

# Sidebar
st.sidebar.header("⚙️ Configuration")
available_models = get_available_ollama_models()
st.sidebar.info(f"🤖 Found {len(available_models)} Ollama model(s)")

selected_model = st.sidebar.selectbox("Select Model", available_models, index=0, key="model_select")
top_k = st.sidebar.slider("🔍 Top K Results", 1, 10, 3, key="topk_slider")
timeout = st.sidebar.slider("⏱️ Query Timeout (s)", 30, 180, 90, key="timeout_slider")

st.sidebar.markdown("---")
st.sidebar.header("📂 Index Management")
upload_file = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="csv_uploader")
if upload_file:
    try:
        df = pd.read_csv(upload_file)
        st.sidebar.success(f"📄 Loaded {len(df)} rows")
        if st.sidebar.button("🔨 Build FAISS Index", key="build_index_btn"):
            build_index_from_dataframe(df, embedder)
    except Exception as e:
        st.sidebar.error(f"❌ Error: {e}")

if os.path.exists(FAISS_INDEX_FILE):
    index, metadata = load_index()
    st.sidebar.success(f"✅ Index loaded: {index.ntotal} vectors")
    if st.sidebar.button("🔄 Rebuild Index", key="rebuild_index_btn"):
        if upload_file:
            df = pd.read_csv(upload_file)
            build_index_from_dataframe(df, embedder)
        else:
            st.sidebar.warning("Upload CSV first")
else:
    st.sidebar.info("⬆️ Upload CSV to build index")
    index, metadata = None, []

st.sidebar.markdown("---")
st.sidebar.header("🎯 Filters")
filter_queue = st.sidebar.selectbox("Queue Filter", ["All"] + list(set([m.get("queue", "") for m in metadata if m.get("queue")])), key="filter_queue")
filter_priority = st.sidebar.selectbox("Priority Filter", ["All", "High", "Medium", "Low"], key="filter_priority")
filter_language = st.sidebar.selectbox("Language Filter", ["All"] + list(set([m.get("language", "") for m in metadata if m.get("language")])), key="filter_language")
filters = {}
if filter_queue != "All": filters["queue"] = filter_queue
if filter_priority != "All": filters["priority"] = filter_priority
if filter_language != "All": filters["language"] = filter_language

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Query Assistant", "📊 Analytics Dashboard", "🕘 History"])

# TAB 1
with tab1:
    st.header("🧠 AI-Powered Support Assistant")

    col1, col2 = st.columns([2, 1])
    with col1:
        subject = st.text_input("📧 Subject", placeholder="Brief ticket subject...")
        body = st.text_area("📝 Description", height=180, placeholder="Detailed issue description...")
    with col2:
        requester_email = st.text_input("📧 Requester Email (optional)", placeholder="user@company.com", key="req_email")
        st.info("**Features:**\n\n🎯 Sentiment Analysis\n🏷️ Auto Topic Tagging\n🌐 Language Detection\n⚡ Priority Boosting")

    if st.button("🚀 Retrieve & Analyze", type="primary", key="analyze_btn"):
        if not subject.strip() and not body.strip():
            st.warning("⚠️ Please enter a subject or body")
        elif not index:
            st.error("❌ No FAISS index found. Build one first.")
        else:
            start_time = time.time()

            combined_text = f"{subject} {body}"
            sentiment, polarity = detect_sentiment(combined_text)
            detected_lang = detect_language(combined_text)
            topics = extract_topics(combined_text)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                sentiment_emoji = "😊" if sentiment == "Positive" else "😟" if sentiment == "Negative" else "😐"
                st.metric("Sentiment", f"{sentiment} {sentiment_emoji}", f"{polarity:.2f}")
            with col_b:
                st.metric("Language", detected_lang.upper())
            with col_c:
                st.metric("Topics", ", ".join(topics[:2]))

            query_text = f"Subject: {subject}\nBody: {body}"
            results = query_faiss(index, metadata, query_text, top_k=top_k, filters=filters)

            gap_flag = (not results) or (len(body.strip()) < 10)

            with st.expander("🧩 Retrieved Context", expanded=True):
                context_texts = []
                if not results:
                    st.warning("No matching tickets found. Try different filters.")
                else:
                    for i, r in enumerate(results, 1):
                        ctx = (
                            f"**Example {i}:**\n"
                            f"**Subject:** {r['subject']}\n"
                            f"**Body:** {r['body'][:150]}...\n"
                            f"**Answer:** {r['answer'][:150]}...\n"
                            f"**Queue:** {r['queue']} | **Priority:** {r['priority']} | **Language:** {r['language']}\n"
                        )
                        st.markdown(ctx)
                        st.markdown("---")
                        context_texts.append(
                            f"Example {i}:\n"
                            f"Subject: {r['subject']}\n"
                            f"Body: {r['body']}\n"
                            f"Here is how we solved a similar case: {r['answer']}\n"
                            f"Queue: {r.get('queue','')}\n"
                            f"Priority: {r.get('priority','')}\n"
                            f"Language: {r.get('language','')}\n"
                        )

            # Append KB_Gap_Fill overlays from previous filled gaps
            try:
                if os.path.exists(HISTORY_FILE):
                    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                        h = json.load(f)
                    overlays = []
                    for r in h:
                        kg = r.get("knowledge_gap", {})
                        if isinstance(kg, dict) and kg.get("kb_fill_text"):
                            if set(r.get("topics", [])).intersection(set(topics)):
                                overlays.append({"ticket_id": r.get("ticket_id"), "text": kg["kb_fill_text"]})
                    if overlays:
                        for g in overlays[:3]:
                            context_texts.append(f"KB_Gap_Fill (from ticket {g['ticket_id']}):\n{g['text']}\n")
            except Exception:
                pass

            # Query Ollama
            if results or context_texts:
                retrieved_context = "\n---\n".join(context_texts)
                prompt = (
                    "You are a senior support agent. Review retrieved KB answers and draft a clear customer reply.\n"
                    "Use the KB answers as primary guidance; if multiple apply, synthesize succinctly.\n"
                    "If any KB_Gap_Fill entries are present, treat them as authoritative updates and quote them when relevant.\n"
                    "Classify priority and assign a practical category for routing.\n\n"
                    "Output ONLY valid JSON (no prose before/after):\n"
                    "{\n"
                    '  "priority": "High" | "Medium" | "Low",\n'
                    '  "category": "short-category",\n'
                    '  "reply": "final customer-ready answer"\n'
                    "}\n\n"
                    "### Retrieved Knowledge (use these KB answers) ###\n"
                    f"{retrieved_context}\n\n"
                    "### New Ticket ###\n"
                    f"Subject: {subject}\n"
                    f"Body: {body}\n\n"
                    f"Detected Sentiment: {sentiment} ({polarity:.2f})\n"
                    f"Topics: {', '.join(topics)}\n\n"
                    "Constraints:\n"
                    "- Keep reply specific to the ticket.\n"
                    "- If info is missing, ask for exact missing details (screenshots, steps, timestamps).\n"
                    "- Do not invent policies; stick to KB_Answer content where possible.\n For any info like tel_number and other personal stuff just Give a random number and mail id of this company called help@knowledgeengine.com\n"
                )
                with st.spinner(f"🤖 Querying {selected_model}..."):
                    raw_answer = run_ollama_cli(selected_model, prompt, timeout=timeout)
            else:
                raw_answer = '{"priority":"Low","category":"General","reply":"Acknowledged. Please share more details so we can assist."}'

            elapsed = time.time() - start_time

            parsed_priority, parsed_reply, parsed_category = parse_llm_response(raw_answer)
            if parsed_priority:
                original_priority = parsed_priority
                parsed_priority = adjust_priority_by_sentiment(parsed_priority, sentiment, polarity)
                if original_priority != parsed_priority:
                    st.info(f"🎯 Priority boosted from {original_priority} to {parsed_priority} due to negative sentiment")

            st.markdown("### 🎯 Analysis Results")
            col_x, col_y, col_z = st.columns(3)
            with col_x:
                priority_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                st.metric("Priority", f"{priority_color.get(parsed_priority, '⚪')} {parsed_priority or 'Unknown'}")
            with col_y:
                st.metric("Category", parsed_category or "General")
            with col_z:
                st.metric("Response Time", f"{elapsed:.1f}s")

            if not parsed_reply:
                parsed_reply = "Thanks for reaching out. Could you share more details (steps to reproduce, screenshots, timestamps) so we can assist quickly?"
            st.markdown("**Suggested Reply:**")
            st.info(parsed_reply)

            with st.expander("🔍 Raw LLM Output"):
                st.code(raw_answer, language="json")

            # Save history with short ID, email, knowledge_gap
            history = []
            if os.path.exists(HISTORY_FILE):
                try:
                    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                        history = json.load(f)
                except Exception:
                    history = []

            ticket_id = short_id(9)
            knowledge_gap = {
                "flagged": bool(gap_flag),
                "gap_reason": "No good retrievals" if gap_flag else "",
                "kb_fill_text": "",
                "filled_by": "",
                "filled_at": None
            }

            record = {
                "ticket_id": ticket_id,
                "ticket_status": "Open",
                "timestamp": time.time(),
                "updated_at": time.time(),
                "subject": subject,
                "body": body,
                "requester_email": requester_email.strip() if requester_email else "",
                "sentiment": sentiment,
                "polarity": polarity,
                "language": detected_lang,
                "topics": topics,
                "retrieved": results,
                "raw_answer": raw_answer,
                "parsed_priority": parsed_priority,
                "parsed_category": parsed_category,
                "parsed_reply": parsed_reply,
                "response_time": elapsed,
                "model": selected_model,
                "knowledge_gap": knowledge_gap,
                "severity": "S3",
                "source": "web"
            }
            history.append(record)
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            st.success(f"✅ Saved to history • Ticket ID: {ticket_id}")

            # Slack notification
            try:
                priority_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(parsed_priority, "⚪")
                sentiment_emoji = "😊" if sentiment == "Positive" else "😟" if sentiment == "Negative" else "😐"
                short_reply = parsed_reply.strip()
                if len(short_reply) > 180:
                    short_reply = short_reply[:177] + "..."

                admin_view_url  = f"{ADMIN_APP_URL}/?ticket_id={ticket_id}"
                admin_open_url  = f"{ADMIN_APP_URL}/?ticket_id={ticket_id}&action=open"
                admin_close_url = f"{ADMIN_APP_URL}/?ticket_id={ticket_id}&action=close"
                admin_fill_url  = f"{ADMIN_APP_URL}/?ticket_id={ticket_id}&action=fill_gap"

                base_text = (
                    f"{APP_NAME}: {priority_emoji} {parsed_priority or 'Unknown'} | "
                    f"{sentiment_emoji} {sentiment} ({polarity:.2f}) | "
                    f"Category: {parsed_category or 'General'}"
                )

                email_line = f" • *Email:* {record['requester_email']}" if record["requester_email"] else ""
                blocks = [
                    {"type": "header", "text": {"type": "plain_text", "text": "New Ticket Analysis"}},
                    {"type": "section", "fields": [
                        {"type": "mrkdwn", "text": f"*Subject:*\n{subject or '-'}"},
                        {"type": "mrkdwn", "text": f"*Priority:*\n{parsed_priority or 'Unknown'}"},
                        {"type": "mrkdwn", "text": f"*Category:*\n{parsed_category or 'General'}"},
                        {"type": "mrkdwn", "text": f"*Sentiment:*\n{sentiment} ({polarity:.2f})"},
                        {"type": "mrkdwn", "text": f"*Language:*\n{detected_lang.upper()}"},
                        {"type": "mrkdwn", "text": f"*Topics:*\n{', '.join(topics) if topics else '-'}"},
                    ]},
                    {"type": "section", "text": {"type": "mrkdwn", "text": f"*Suggested Reply:*\n{short_reply or '-'}"}},
                    {"type": "context","elements":[
                        {"type":"mrkdwn","text": f"*Ticket ID:* `{ticket_id}` • *Status:* Open{email_line}"}
                    ]},
                    {"type":"actions","elements":[
                        {"type":"button","text":{"type":"plain_text","text":"Open in Admin"}, "url": admin_view_url},
                        {"type":"button","text":{"type":"plain_text","text":"Mark Open"}, "url": admin_open_url},
                        {"type":"button","text":{"type":"plain_text","text":"Mark Closed"}, "url": admin_close_url},
                        {"type":"button","text":{"type":"plain_text","text":"Fill Gap"}, "url": admin_fill_url},
                    ]}
                ]

                should_notify = (parsed_priority == "High") or (sentiment == "Negative")
                if knowledge_gap["flagged"]:
                    post_to_slack(f"🕳️ Knowledge Gap flagged for ticket {ticket_id}", blocks)
                elif should_notify:
                    post_to_slack(base_text, blocks)
            except Exception as e:
                st.warning(f"Slack block build failed: {e}")

# TAB 2
with tab2:
    st.header("📊 Analytics Dashboard")
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                hist_data = json.load(f)
            if hist_data:
                hist_df = pd.DataFrame(hist_data)
                hist_df["datetime"] = pd.to_datetime(hist_df["timestamp"], unit="s")
                for col in ["sentiment","parsed_category","response_time","polarity","language","topics","ticket_status","requester_email","severity","source"]:
                    if col not in hist_df.columns:
                        if col in ["sentiment"]: hist_df[col] = "Neutral"
                        elif col in ["parsed_category"]: hist_df[col] = "General"
                        elif col in ["response_time","polarity"]: hist_df[col] = 0.0
                        elif col in ["language"]: hist_df[col] = "en"
                        elif col in ["topics"]: hist_df[col] = hist_df.apply(lambda x: ["General"], axis=1)
                        elif col in ["ticket_status"]: hist_df[col] = "Open"
                        elif col in ["requester_email","severity","source"]: hist_df[col] = ""

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📨 Total Tickets", len(hist_df))
                with col2:
                    st.metric("⏱️ Avg Response Time", f"{hist_df['response_time'].mean():.1f}s")
                with col3:
                    high_count = (hist_df["parsed_priority"] == "High").sum() if "parsed_priority" in hist_df else 0
                    st.metric("🔴 High Priority", int(high_count))
                with col4:
                    neg_count = (hist_df["sentiment"] == "Negative").sum() if "sentiment" in hist_df else 0
                    st.metric("😟 Negative Sentiment", int(neg_count))

                st.markdown("---")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("🎯 Priority Distribution")
                    if "parsed_priority" in hist_df:
                        st.bar_chart(hist_df["parsed_priority"].value_counts())
                with col_b:
                    st.subheader("😊 Sentiment Distribution")
                    if "sentiment" in hist_df:
                        st.bar_chart(hist_df["sentiment"].value_counts())

                st.subheader("📈 Ticket Timeline")
                daily_counts = hist_df.groupby(hist_df["datetime"].dt.date).size()
                st.line_chart(daily_counts)

                if "topics" in hist_df:
                    st.subheader("🏷️ Top Topics")
                    all_topics = []
                    for t in hist_df["topics"]:
                        if isinstance(t, list): all_topics.extend(t)
                        elif isinstance(t, str): all_topics.append(t)
                    if all_topics:
                        st.bar_chart(pd.Series(all_topics).value_counts().head(10))
            else:
                st.info("No history data available yet. Process some tickets first!")
        except Exception as e:
            st.error(f"Error loading analytics: {e}")
    else:
        st.info("📭 No history file found. Start querying tickets to see analytics!")

# TAB 3
with tab3:
    st.header("🕘 Ticket History")
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            hist = json.load(f)
        if hist:
            hist_df = pd.DataFrame(hist)
            hist_df["datetime"] = pd.to_datetime(hist_df["timestamp"], unit="s")
            required_cols_defaults = {
                "ticket_id": "",
                "ticket_status": "Open",
                "sentiment": "Neutral",
                "parsed_category": "General",
                "response_time": 0.0,
                "polarity": 0.0,
                "language": "en",
                "topics": ["General"],
                "requester_email": "",
                "severity": "S3",
                "source": "web"
            }
            for col, default_val in required_cols_defaults.items():
                if col not in hist_df.columns:
                    hist_df[col] = default_val

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                filter_hist_priority = st.selectbox("Filter by Priority", ["All","High","Medium","Low"], key="hist_priority")
            with col2:
                filter_hist_sentiment = st.selectbox("Filter by Sentiment", ["All","Positive","Negative","Neutral"], key="hist_sentiment")
            with col3:
                filter_hist_status = st.selectbox("Filter by Status", ["All","Open","Closed"], key="hist_status")
            with col4:
                num_records = st.slider("Show last N records", 5, 50, 10, key="hist_num_records")

            filtered_df = hist_df.copy()
            if filter_hist_priority != "All":
                filtered_df = filtered_df[filtered_df["parsed_priority"] == filter_hist_priority]
            if filter_hist_sentiment != "All":
                filtered_df = filtered_df[filtered_df["sentiment"] == filter_hist_sentiment]
            if filter_hist_status != "All":
                filtered_df = filtered_df[filtered_df["ticket_status"] == filter_hist_status]

            display_df = filtered_df.tail(num_records).sort_values("datetime", ascending=False)
            preferred_cols = ["datetime","ticket_id","ticket_status","subject","parsed_priority","sentiment","parsed_category","response_time","model"]
            available_cols = [c for c in preferred_cols if c in display_df.columns]
            st.dataframe(display_df[available_cols], use_container_width=True)

            st.markdown("---")
            st.subheader("📄 Detailed Records")
            for idx, record in display_df.iterrows():
                with st.expander(f"🎫 {record['datetime'].strftime('%Y-%m-%d %H:%M')} - {record['subject'][:50]}"):
                    col_x, col_y = st.columns(2)
                    with col_x:
                        st.markdown(f"**Ticket ID:** {record.get('ticket_id','-')}")
                        st.markdown(f"**Status:** {record.get('ticket_status','Open')}")
                        st.markdown(f"**Subject:** {record['subject']}")
                        st.markdown(f"**Body:** {record['body'][:200]}...")
                        st.markdown(f"**Priority:** {record.get('parsed_priority','-')}")
                        st.markdown(f"**Category:** {record.get('parsed_category', 'N/A')}")
                    with col_y:
                        st.markdown(f"**Sentiment:** {record.get('sentiment', 'N/A')} ({record.get('polarity', 0):.2f})")
                        st.markdown(f"**Language:** {record.get('language', 'N/A')}")
                        topics_val = record.get('topics', ['N/A'])
                        topics_str = ', '.join(topics_val) if isinstance(topics_val, list) else str(topics_val)
                        st.markdown(f"**Topics:** {topics_str}")
                        st.markdown(f"**Requester Email:** {record.get('requester_email','') or '—'}")
                        st.markdown(f"**Model:** {record.get('model', 'N/A')}")
                        kg = record.get("knowledge_gap", {})
                        if isinstance(kg, dict):
                            st.markdown(f"**Gap Flagged:** {kg.get('flagged', False)}")
                            st.markdown(f"**Gap Reason:** {kg.get('gap_reason','')}")
                            st.markdown(f"**Gap Filled By:** {kg.get('filled_by','')}")
                            if kg.get("filled_at"):
                                st.markdown(f"**Filled At:** {datetime.fromtimestamp(kg['filled_at']).isoformat()}")
                            st.markdown("**KB Gap Fill Text:**")
                            st.code(kg.get("kb_fill_text",""), language="markdown")

            st.markdown("---")
            if st.button("📥 Export History to CSV", key="export_hist_btn"):
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"ticket_history_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No history records yet")
    else:
        st.info("No history file found")
