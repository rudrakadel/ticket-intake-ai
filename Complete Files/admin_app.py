# admin_app.py
"""
Admin Portal for AI Support Assistant Pro

Functions:
- Open/Close ticket status via UI or deep-links (?ticket_id=...&action=open|close|fill_gap)
- Fill Knowledge Gap (kb_fill_text) with audit info
- Optional Slack confirmations
- Basic analytics using Seaborn/Matplotlib
"""

import os, json, time
from datetime import datetime

import streamlit as st
st.set_page_config(page_title="Admin Portal - AI Support Assistant", layout="wide", page_icon="🛠️")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests

HISTORY_FILE = os.getenv("HISTORY_FILE", "conversation_history.json")
SLACK_WEBHOOK_URL = os.getenv(
    "SLACK_WEBHOOK_URL",
    "<Yourwebhook>"
)
ADMIN_NAME = os.getenv("ADMIN_NAME", "Admin")

def post_to_slack(text: str, blocks=None):
    if not SLACK_WEBHOOK_URL:
        return
    payload = {"text": text}
    if blocks:
        payload["blocks"] = blocks
    try:
        r = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=5)
        r.raise_for_status()
    except Exception as e:
        st.warning(f"Slack failed: {e}")

def load_history():
    if not os.path.exists(HISTORY_FILE): return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_history(hist):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2, ensure_ascii=False)

def set_status(ticket_id: str, status: str) -> bool:
    hist = load_history()
    changed = False
    for r in hist:
        if r.get("ticket_id") == ticket_id:
            r["ticket_status"] = status
            r["updated_at"] = time.time()
            changed = True
            break
    if changed:
        save_history(hist)
        blocks = [{"type":"section","text":{"type":"mrkdwn","text": f"*Ticket {ticket_id}* status set to *{status}* by {ADMIN_NAME}"}}]
        post_to_slack(f"Ticket {ticket_id} -> {status}", blocks)
    return changed

def fill_gap(ticket_id: str, kb_text: str) -> bool:
    hist = load_history()
    changed = False
    for r in hist:
        if r.get("ticket_id") == ticket_id:
            if "knowledge_gap" not in r or not isinstance(r["knowledge_gap"], dict):
                r["knowledge_gap"] = {}
            r["knowledge_gap"]["kb_fill_text"] = kb_text.strip()
            r["knowledge_gap"]["filled_by"] = ADMIN_NAME
            r["knowledge_gap"]["filled_at"] = time.time()
            r["updated_at"] = time.time()
            changed = True
            break
    if changed:
        save_history(hist)
        blocks = [{"type":"section","text":{"type":"mrkdwn","text": f"🧩 Gap Fill saved for *{ticket_id}* by {ADMIN_NAME}"}}]
        post_to_slack(f"Gap filled for ticket {ticket_id}", blocks)
    return changed

# Sidebar
st.sidebar.header("Admin Controls")
tid_param = st.sidebar.text_input("Ticket ID", key="adm_tid")
act = st.sidebar.selectbox("Action", ["None","Open","Close","Fill Gap"], key="adm_action")
go = st.sidebar.button("Apply", key="adm_apply")

# Query params deep-link handling (Streamlit >= 1.30)
qs = st.query_params
qp_tid = qs.get("ticket_id", [None])[0] if isinstance(qs.get("ticket_id"), list) else qs.get("ticket_id")
qp_act = qs.get("action", [None])[0] if isinstance(qs.get("action"), list) else qs.get("action")

if qp_tid and qp_act in ("open","close","fill_gap"):
    st.info(f"Deep-link: {qp_act} for {qp_tid}")
    if qp_act == "open":
        if set_status(qp_tid, "Open"):
            st.success(f"Applied Open for {qp_tid}")
    elif qp_act == "close":
        if set_status(qp_tid, "Closed"):
            st.success(f"Applied Closed for {qp_tid}")
    elif qp_act == "fill_gap":
        st.session_state["prefill_tid"] = qp_tid

if go and tid_param and act != "None":
    if act == "Open":
        st.success("Updated" if set_status(tid_param, "Open") else "Ticket not found")
    elif act == "Close":
        st.success("Updated" if set_status(tid_param, "Closed") else "Ticket not found")
    elif act == "Fill Gap":
        st.session_state["prefill_tid"] = tid_param

st.title("🛠️ Admin Portal")
st.caption("Manage ticket statuses, fill knowledge gaps, and view analytics.")

hist = load_history()
if not hist:
    st.info("No tickets yet.")
    st.stop()

df = pd.DataFrame(hist)
df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
if "updated_at" in df.columns:
    df["updated_dt"] = pd.to_datetime(df["updated_at"], unit="s")
else:
    df["updated_dt"] = df["datetime"]

# Filters
col1, col2, col3 = st.columns(3)
with col1:
    f_status = st.selectbox("Filter Status", ["All","Open","Closed"], key="f_status")
with col2:
    f_priority = st.selectbox("Filter Priority", ["All","High","Medium","Low"], key="f_prio")
with col3:
    f_text = st.text_input("Search Subject/Body", key="f_text")

filtered = df.copy()
if f_status != "All":
    filtered = filtered[filtered.get("ticket_status","Open") == f_status]
if f_priority != "All" and "parsed_priority" in filtered:
    filtered = filtered[filtered["parsed_priority"] == f_priority]
if f_text:
    mask = filtered["subject"].str.contains(f_text, case=False, na=False) | filtered["body"].str.contains(f_text, case=False, na=False)
    filtered = filtered[mask]

st.subheader("Tickets")
show_cols = [c for c in ["ticket_id","ticket_status","datetime","subject","parsed_priority","sentiment","parsed_category","requester_email"] if c in filtered.columns]
st.dataframe(filtered[show_cols], use_container_width=True)

st.markdown("---")
st.subheader("Fill Knowledge Gap")
prefill_tid = st.session_state.get("prefill_tid","")
tid = st.text_input("Ticket ID", value=prefill_tid, key="fill_tid")
kb_text = st.text_area("Knowledge Gap Fill (KB text to reuse)", height=220, key="fill_text")
if st.button("Save Gap Fill", key="fill_save"):
    if tid and kb_text.strip():
        st.success("Saved" if fill_gap(tid, kb_text) else "Ticket not found")
    else:
        st.error("Provide Ticket ID and KB text")

st.markdown("---")
st.subheader("Analytics")

fig1, ax1 = plt.subplots(figsize=(5,3))
if "parsed_priority" in filtered:
    sns.countplot(data=filtered, x="parsed_priority", order=["High","Medium","Low"], ax=ax1)
ax1.set_title("Priority Distribution")
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(6,3))
if "sentiment" in filtered and "ticket_status" in filtered:
    sns.countplot(data=filtered, x="sentiment", hue="ticket_status", order=["Negative","Neutral","Positive"], ax=ax2)
ax2.set_title("Sentiment by Status")
st.pyplot(fig2)

tmp = filtered.copy()
if "ticket_status" in tmp and "updated_dt" in tmp:
    closed = tmp[tmp["ticket_status"]=="Closed"]
    if not closed.empty:
        delta = (closed["updated_dt"] - closed["datetime"]).dt.total_seconds()/3600.0
        fig3, ax3 = plt.subplots(figsize=(6,3))
        sns.histplot(delta, bins=20, ax=ax3, kde=True)
        ax3.set_title("Time to Close (hours)")
        st.pyplot(fig3)

st.markdown("---")
st.caption("Admin Portal v1 • Uses the same conversation_history.json as the main app.")
