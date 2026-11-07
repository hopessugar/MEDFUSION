"""MedFusion-RAFT Research Interface.

Interactive clinical question answering, grounding verification, and evaluation
dashboard built with Streamlit.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="MedFusion-RAFT Research Interface",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom medical styling
st.markdown(
    """
    <style>
    .main-header { font-size: 2.1rem; font-weight: 700; color: #0f4c81; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #4a5568; margin-bottom: 1.2rem; }
    .disclaimer-box {
        background-color: #fff3cd; border: 1.5px solid #ffeeba; color: #856404;
        padding: 0.75rem 1.2rem; border-radius: 8px; font-weight: 600; margin-bottom: 1.5rem;
    }
    .badge-accept { background-color: #28a745; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold; }
    .badge-revise { background-color: #ffc107; color: #212529; padding: 4px 12px; border-radius: 12px; font-weight: bold; }
    .badge-abstain { background-color: #dc3545; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold; }
    .evidence-card { background-color: #f8fafc; border-left: 4px solid #0f4c81; padding: 0.8rem; margin-bottom: 0.6rem; border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Prominent Research Prototype Disclaimer
st.markdown(
    '<div class="disclaimer-box">⚠️ RESEARCH PROTOTYPE - Not for clinical use or medical decision-making.</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="main-header">MedFusion-RAFT Research Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Evidence-Grounded Medical Question Answering with Claim Verification and Selective Abstention</div>',
    unsafe_allow_html=True,
)

# Sidebar Configuration
st.sidebar.header("⚙️ Pipeline Configuration")
api_url = st.sidebar.text_input("Backend API URL", value="http://localhost:8000")

experiment_options = [
    "E1_base", "E2_bm25", "E3_dense", "E4_hybrid",
    "E5_raft", "E6_sft_raft", "E7_verify", "E8_abstain", "E9_full"
]
selected_exp = st.sidebar.selectbox("Experiment Configuration", experiment_options, index=8)

use_retrieval = st.sidebar.toggle("Enable Retrieval Augmentation", value=True)
use_verification = st.sidebar.toggle("Enable Claim Verification", value=True)
use_abstention = st.sidebar.toggle("Enable Selective Abstention", value=True)
mock_mode = st.sidebar.checkbox("Use Mock Providers (CPU Testing)", value=False)

# Helper function for API calls
def call_api(endpoint: str, method: str = "GET", payload: Optional[dict] = None) -> Optional[dict]:
    """Execute HTTP call to the backend API with graceful offline fallback."""
    full_url = f"{api_url.rstrip('/')}/api/v1/{endpoint.lstrip('/')}"
    try:
        with httpx.Client(timeout=45.0) as client:
            if method.upper() == "POST":
                resp = client.post(full_url, json=payload or {})
            else:
                resp = client.get(full_url)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        st.sidebar.error(f"Backend API offline or unreachable: {exc}")
        return None

# Tabs for primary workflows
tab_ask, tab_batch, tab_health = st.tabs(["🩺 Clinical Query", "📊 Benchmark Evaluation", "🔍 System Health"])

# TAB 1: Single Question Analysis
with tab_ask:
    st.subheader("Clinical Inquiry")
    default_q = "What is the recommended first-line treatment for uncomplicated primary hypertension?"
    question_input = st.text_area("Enter medical query or clinical scenario:", value=default_q, height=100)
    specialty_input = st.text_input("Specialty tag (optional):", value="Cardiology")

    if st.button("Generate & Verify Answer", type="primary"):
        with st.spinner("Executing retrieval, inference, and claim verification..."):
            request_data = {
                "question": question_input,
                "specialty": specialty_input or None,
                "use_retrieval": use_retrieval,
                "use_verification": use_verification,
                "use_abstention": use_abstention,
                "experiment_id": selected_exp,
            }
            res = call_api("ask", method="POST", payload=request_data)

            if res:
                st.markdown("### Answer Synthesis")
                decision = res.get("decision", "accept").lower()
                decision_badge = {
                    "accept": '<span class="badge-accept">DECISION: ACCEPT</span>',
                    "revise": '<span class="badge-revise">DECISION: REVISE</span>',
                    "abstain": '<span class="badge-abstain">DECISION: ABSTAIN</span>',
                }.get(decision, f"<span>{decision.upper()}</span>")

                col_badge, col_conf, col_time = st.columns([2, 2, 2])
                with col_badge:
                    st.markdown(decision_badge, unsafe_allow_html=True)
                with col_conf:
                    st.metric("Confidence", f"{res.get('confidence', 1.0):.2f}")
                with col_time:
                    st.metric("Latency", f"{res.get('processing_time_ms', 0):.1f} ms")

                st.info(res.get("answer_text", ""))

                # Verification and Claims Section
                claims = res.get("claims", [])
                if claims:
                    st.markdown("### 🔬 Claim-Level Verification Table")
                    df_claims = pd.DataFrame(claims)
                    st.dataframe(df_claims, use_container_width=True)

                # Evidence Passages
                passages = res.get("evidence_passages", [])
                if passages:
                    st.markdown("### 📚 Retrieved Evidence Passages")
                    for p in passages:
                        with st.expander(f"Passage: {p.get('passage_id')} (Score: {p.get('score', 0):.3f})"):
                            st.write(p.get("text"))
                            st.caption(f"Source: {p.get('source', 'Unknown')} | Section: {p.get('section', 'General')}")
            else:
                st.warning(
                    "⚠️ Could not reach the API. Start the backend with:\n\n"
                    "```powershell\npython -m uvicorn backend.app:create_app --port 8000\n```"
                )

# TAB 2: Batch & Dataset Evaluation
with tab_batch:
    st.subheader("Batch Question Evaluation")
    uploaded_file = st.file_uploader("Upload benchmark dataset (.json, .jsonl, .csv)", type=["json", "jsonl", "csv"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_data = pd.read_csv(uploaded_file)
            else:
                raw_bytes = uploaded_file.read()
                raw_str = raw_bytes.decode("utf-8")
                if uploaded_file.name.endswith(".jsonl"):
                    items = [json.loads(line) for line in raw_str.splitlines() if line.strip()]
                else:
                    items = json.loads(raw_str)
                df_data = pd.DataFrame(items)

            st.write(f"Loaded dataset with **{len(df_data)}** rows:")
            st.dataframe(df_data.head(5), use_container_width=True)

            q_col = st.selectbox("Select question column:", df_data.columns)

            if st.button("Run Batch Pipeline Evaluation"):
                with st.spinner("Processing evaluation batch..."):
                    batch_questions = [
                        {"question": str(row[q_col]), "experiment_id": selected_exp}
                        for _, row in df_data.iterrows() if str(row[q_col]).strip()
                    ]
                    batch_payload = {"questions": batch_questions, "experiment_id": selected_exp}
                    batch_res = call_api("batch", method="POST", payload=batch_payload)

                    if batch_res:
                        results = batch_res.get("results", [])
                        st.success(f"Batch completed in {batch_res.get('total_time_ms', 0):.1f} ms!")

                        res_records = [
                            {
                                "Question": q["question"],
                                "Answer": r.get("answer_text"),
                                "Decision": r.get("decision"),
                                "Confidence": r.get("confidence"),
                                "Latency (ms)": r.get("processing_time_ms"),
                            }
                            for q, r in zip(batch_questions, results)
                        ]
                        df_res = pd.DataFrame(res_records)
                        st.dataframe(df_res, use_container_width=True)

                        # Metric summary
                        st.markdown("#### Summary Metrics")
                        col_acc, col_rev, col_abs = st.columns(3)
                        col_acc.metric("Accepted", sum(df_res["Decision"] == "accept"))
                        col_rev.metric("Revise", sum(df_res["Decision"] == "revise"))
                        col_abs.metric("Abstained", sum(df_res["Decision"] == "abstain"))

                        # Download options
                        csv_data = df_res.to_csv(index=False).encode("utf-8")
                        st.download_button("Download CSV Results", data=csv_data, file_name="batch_evaluation_results.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error reading dataset file: {e}")

# TAB 3: System Health & Diagnostics
with tab_health:
    st.subheader("Service Diagnostics")
    health_data = call_api("health")
    if health_data:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Status", health_data.get("status", "unknown").upper())
        c2.metric("Models Loaded", str(health_data.get("models_loaded", False)))
        c3.metric("GPU Accelerated", str(health_data.get("gpu_available", False)))
        c4.metric("Uptime", f"{health_data.get('uptime_seconds', 0):.1f} s")
    else:
        st.info("API is currently offline. Launch backend to see real-time diagnostics.")
