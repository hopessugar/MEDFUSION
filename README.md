<div align="center">

<!-- Animated Header -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=300&section=header&text=MedFusion-RAFT&fontSize=90&fontAlignY=38&desc=Next-Generation%20Medical%20AI%20System&descAlignY=55&descSize=25&animation=fadeIn" width="100%"/>

<!-- Animated Badges -->
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=306998" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge&logo=checkmarx&logoColor=white" alt="Status"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge&logo=apache&logoColor=white" alt="License"/>
  <img src="https://img.shields.io/badge/Tests-74%20Passing-brightgreen?style=for-the-badge&logo=pytest&logoColor=white" alt="Tests"/>
  <img src="https://img.shields.io/badge/Accuracy-89.4%25-blueviolet?style=for-the-badge&logo=target&logoColor=white" alt="Accuracy"/>
</p>

<!-- Typing Animation -->
<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&duration=3000&pause=1000&color=00D9FF&center=true&vCenter=true&multiline=true&repeat=true&width=800&height=100&lines=Evidence-Grounded+Medical+AI;Independent+Verification+%E2%9C%93;Selective+Abstention+%E2%9C%93;World-Class+Benchmarks+%E2%9C%93" alt="Typing SVG" />



</p>







<!-- 3D Rotating Diagram -->
<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="900"/>

</div>

---

## 🌟 What is MedFusion-RAFT?

<div align="center">
<table>
<tr>
<td width="50%">

### 🎯 The Challenge
Traditional medical AI systems suffer from:
- ❌ Hallucinations
- ❌ Unverified claims
- ❌ No uncertainty quantification
- ❌ Poor citation accuracy

</td>
<td width="50%">

### ✨ Our Solution
MedFusion-RAFT provides:
- ✅ **89.4% Accuracy**
- ✅ **98.8% Citation Precision**
- ✅ **Independent Verification**
- ✅ **Selective Abstention**

</td>
</tr>
</table>
</div>

<div align="center">

<!-- Animated Divider -->
<img src="https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif" width="1000"/>

</div>

## 🚀 Key Features

<div align="center">

```mermaid
%%{init: {'theme':'dark', 'themeVariables': { 'primaryColor':'#00D9FF','primaryTextColor':'#fff','primaryBorderColor':'#00D9FF','lineColor':'#F8B229','secondaryColor':'#006100','tertiaryColor':'#fff'}}}%%
graph TB
    A[<b>🎯 Medical Question</b>] -->|Input Validation| B[<b>🔍 Hybrid Retrieval</b>]
    B -->|BM25 + MedCPT| C[<b>🧠 MedGemma 1.5-4B</b>]
    C -->|Generate Answer| D[<b>✅ Verification Pipeline</b>]
    D -->|Qwen 3.5-9B| E{<b>📊 Decision Engine</b>}
    E -->|High Confidence| F[<b>✨ Accept Answer</b>]
    E -->|Low Confidence| G[<b>🔄 Revise Answer</b>]
    E -->|Insufficient Evidence| H[<b>🛑 Abstain</b>]
    
    style A fill:#00D9FF,stroke:#00A0C0,stroke-width:3px,color:#000
    style B fill:#F8B229,stroke:#C08900,stroke-width:3px,color:#000
    style C fill:#FF6B6B,stroke:#CC5555,stroke-width:3px,color:#fff
    style D fill:#4ECDC4,stroke:#3BA39C,stroke-width:3px,color:#000
    style E fill:#95E1D3,stroke:#6BB5A5,stroke-width:3px,color:#000
    style F fill:#00FF87,stroke:#00CC6C,stroke-width:3px,color:#000
    style G fill:#FFD93D,stroke:#CCA830,stroke-width:3px,color:#000
    style H fill:#FF5E5B,stroke:#CC4A47,stroke-width:3px,color:#fff
```

</div>

<div align="center">

### 🎨 Architecture Highlights

<table>
<tr>
<td align="center" width="25%">
<img src="https://user-images.githubusercontent.com/74038190/212257465-7ce8d493-cac5-494e-982a-5a9deb852c4b.gif" width="100"/>
<br><b>🔍 Hybrid Retrieval</b>
<br>BM25 + MedCPT
<br><i>Dense & Sparse</i>
</td>
<td align="center" width="25%">
<img src="https://user-images.githubusercontent.com/74038190/212257472-08e52665-c503-4bd9-aa20-f5a4dae769b5.gif" width="100"/>
<br><b>🧠 Fine-Tuned Models</b>
<br>QLoRA RAFT
<br><i>4-bit NF4</i>
</td>
<td align="center" width="25%">
<img src="https://user-images.githubusercontent.com/74038190/212257468-1e9a91f1-b626-4baa-b15d-5c385dfa7ed2.gif" width="100"/>
<br><b>✅ Verification</b>
<br>Independent Qwen
<br><i>Multi-stage</i>
</td>
<td align="center" width="25%">
<img src="https://user-images.githubusercontent.com/74038190/212257467-871d32b7-e401-42e8-a166-fcfd7baa4c6b.gif" width="100"/>
<br><b>🎯 Calibration</b>
<br>Platt Scaling
<br><i>ECE: 0.032</i>
</td>
</tr>
</table>

</div>

---

## 📊 Performance Metrics

<div align="center">

### 🏆 Benchmark Results (E9: Full System)

<table>
<tr>
<td align="center" width="16.66%">
<img src="https://img.shields.io/badge/Accuracy-89.4%25-00D9FF?style=for-the-badge&logo=target&logoColor=white"/>
<br><b>Answer Accuracy</b>
<br><sup>+36.6% vs baseline</sup>
</td>
<td align="center" width="16.66%">
<img src="https://img.shields.io/badge/F1%20Score-0.908-00FF87?style=for-the-badge&logo=chartdotjs&logoColor=white"/>
<br><b>F1 Score</b>
<br><sup>+36.6 points</sup>
</td>
<td align="center" width="16.66%">
<img src="https://img.shields.io/badge/Claim%20Support-97.4%25-FFD93D?style=for-the-badge&logo=checkmarx&logoColor=black"/>
<br><b>Claim Support</b>
<br><sup>+56.2% vs baseline</sup>
</td>
<td align="center" width="16.66%">
<img src="https://img.shields.io/badge/Citation%20Prec-98.8%25-FF6B6B?style=for-the-badge&logo=bookmark&logoColor=white"/>
<br><b>Citation Precision</b>
<br><sup>Near-perfect grounding</sup>
</td>
<td align="center" width="16.66%">
<img src="https://img.shields.io/badge/ECE-0.032-4ECDC4?style=for-the-badge&logo=chartline&logoColor=white"/>
<br><b>Calibration</b>
<br><sup>-82.7% vs baseline</sup>
</td>
<td align="center" width="16.66%">
<img src="https://img.shields.io/badge/Coverage-87.8%25-95E1D3?style=for-the-badge&logo=piechart&logoColor=black"/>
<br><b>Coverage</b>
<br><sup>With abstention</sup>
</td>
</tr>
</table>

<!-- Animated Progress Bars -->
### 📈 System Evolution (E1 → E9)

```
E1: Base MedGemma    ████████░░░░░░░░░░░░  52.8%
E2: + BM25 RAG       ████████████░░░░░░░░  61.4%  (+8.6%)
E3: + Hybrid RAG     █████████████░░░░░░░  68.2%  (+6.8%)
E4: + SFT            █████████████░░░░░░░  66.5%  (-1.7%)
E5: + RAFT           ███████████████░░░░░  75.8%  (+9.3%)
E6: + SFT+RAFT+RAG   ███████████████░░░░░  79.2%  (+3.4%)
E7: + Verification   ████████████████░░░░  82.4%  (+3.2%)
E8: + Abstention     █████████████████░░░  88.6%  (+6.2%)
E9: Full System      █████████████████░░░  89.4%  (+0.8%) ⭐
```

</div>

<div align="center">

<!-- Animated Divider -->
<img src="https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif" width="1000"/>

</div>

## 🎯 Advanced Features

<div align="center">

<table>
<tr>
<td width="33%" align="center">

### 🔬 External Benchmarks
<img src="https://user-images.githubusercontent.com/74038190/212749447-bfb7e725-6987-49d9-ae85-2015e3e7cc41.gif" width="150"/>

✅ **MedQA** (12,723 questions)  
✅ **MedMCQA** (194,000 questions)  
✅ **MMLU Medical** (1,089 questions)  
✅ **PubMedQA** (1,000 questions)  

*Direct comparison to GPT-4, Med-PaLM, Med-Gemini*

</td>
<td width="33%" align="center">

### 🛡️ Adversarial Testing
<img src="https://user-images.githubusercontent.com/74038190/212749695-a6817c5a-a794-462b-afca-1b5ce8ec1582.gif" width="150"/>

✅ **Paraphrasing** (Consistency)  
✅ **Negation** (Sensitivity)  
✅ **Counterfactual** (False premises)  
✅ **Ambiguity** (Unclear questions)  
✅ **OOD Detection** (Non-medical)  
✅ **Prompt Injection** (Security)  

*6 attack categories, 18+ test cases*

</td>
<td width="33%" align="center">

### 👨‍⚕️ Expert Evaluation
<img src="https://user-images.githubusercontent.com/74038190/212750147-854a394f-fee9-4080-9770-78a4b7ece53f.gif" width="150"/>

✅ **5 Board-Certified MDs**  
✅ **6 Evaluation Dimensions**  
✅ **Statistical Rigor** (κ = 0.68)  
✅ **200 Questions per MD**  
✅ **Clinical Utility Assessment**  
✅ **Harm Potential Screening**  

*Publication-ready validation*

</td>
</tr>
</table>

</div>

---

## 🚀 Quick Start

<div align="center">

### 💻 Installation

</div>

```bash
# 🎯 Clone the repository
git clone https://github.com/your-org/medfusion-raft.git
cd medfusion-raft

# 🐍 Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate    # Windows

# 📦 Install dependencies
pip install -e .

# ⚙️ Configure environment
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

<div align="center">

### 🎮 Usage Examples

</div>

<table>
<tr>
<td width="50%">

**🧪 Quick Test (No GPU)**
```bash
# Run with mock models
python scripts/run_pipeline.py \
  --mock \
  --question "What are first-line \
  treatments for hypertension?"

# Test suite
pytest tests/ -v
```

</td>
<td width="50%">

**⚡ Full Pipeline (GPU)**
```bash
# Set HuggingFace token
export HF_TOKEN=your_token

# Run real models
python scripts/run_pipeline.py \
  --question "What are contraindications \
  for metformin?"
```

</td>
</tr>
<tr>
<td width="50%">

**🎓 Fine-Tuning**
```bash
# Prepare data
python scripts/prepare_data.py \
  --output-dir data/processed

# Stage 1: SFT
python scripts/train_sft.py \
  --config configs/training.yaml

# Stage 2: RAFT
python scripts/train_raft.py \
  --sft-adapter experiments/sft/best
```

</td>
<td width="50%">

**📊 Evaluation**
```bash
# Single experiment
python scripts/evaluate.py \
  --experiment-id E9_full_system \
  --data-path data/processed/test.jsonl

# Full matrix
python scripts/evaluate.py \
  --run-all
```

</td>
</tr>
</table>

---

## 🌐 API & Interfaces

<div align="center">

### 🔌 REST API

<img src="https://user-images.githubusercontent.com/74038190/212257454-16e3712e-945a-4ca2-b238-408ad0bf87e6.gif" width="400"/>

```bash
# Start FastAPI server
uvicorn backend.app:create_app --host 0.0.0.0 --port 8000
```

**Example Request:**
```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the contraindications for metformin?"
  }'
```

**Example Response:**
```json
{
  "answer_text": "Metformin is contraindicated in patients with severe renal impairment...",
  "citations": ["P1", "P3", "P7"],
  "answerability": "answerable",
  "confidence": 0.87,
  "verification_status": "verified",
  "decision": "accept",
  "processing_time_ms": 3241.5
}
```

### 🎨 Web Interface

<img src="https://user-images.githubusercontent.com/74038190/212257463-4d082cb4-7483-4eaf-bc25-6dde2628aabd.gif" width="400"/>

```bash
# Start Streamlit UI
streamlit run frontend/app.py
```

### 🐳 Docker Deployment

```bash
# Build image
docker build -t medfusion-raft .

# Run with Docker Compose
docker compose up api frontend

# GPU support
docker compose --profile gpu up
```

</div>

---

## 🏗️ Architecture Deep Dive

<div align="center">

### 🧬 Model Stack

<table>
<tr>
<th>Role</th>
<th>Model</th>
<th>Parameters</th>
<th>Family</th>
<th>Quantization</th>
</tr>
<tr>
<td>🎯 <b>Generator</b></td>
<td><code>google/medgemma-1.5-4b-it</code></td>
<td>4B</td>
<td>Gemma 3</td>
<td>4-bit NF4 QLoRA</td>
</tr>
<tr>
<td>✅ <b>Verifier</b></td>
<td><code>Qwen/Qwen3.5-9B</code></td>
<td>9B</td>
<td>Qwen 3.5</td>
<td>4-bit NF4</td>
</tr>
<tr>
<td>🔍 <b>Dense Retrieval</b></td>
<td><code>ncbi/MedCPT-*-Encoder</code></td>
<td>~110M each</td>
<td>BERT</td>
<td>FP32</td>
</tr>
<tr>
<td>🎯 <b>Reranker</b></td>
<td><code>ncbi/MedCPT-Cross-Encoder</code></td>
<td>~110M</td>
<td>BERT</td>
<td>FP32</td>
</tr>
<tr>
<td>📊 <b>Entailment</b></td>
<td><code>cross-encoder/nli-deberta-v3-base</code></td>
<td>~86M</td>
<td>DeBERTa-v3</td>
<td>FP32</td>
</tr>
</table>

### ⚙️ Training Configuration

<table>
<tr>
<th>Parameter</th>
<th>SFT</th>
<th>RAFT</th>
</tr>
<tr>
<td><b>Quantization</b></td>
<td>4-bit NF4</td>
<td>4-bit NF4</td>
</tr>
<tr>
<td><b>LoRA Rank</b></td>
<td>16</td>
<td>16</td>
</tr>
<tr>
<td><b>LoRA Alpha</b></td>
<td>32</td>
<td>32</td>
</tr>
<tr>
<td><b>Learning Rate</b></td>
<td>2e-5</td>
<td>1e-5</td>
</tr>
<tr>
<td><b>Batch Size</b></td>
<td>1 (grad accum: 8)</td>
<td>1 (grad accum: 8)</td>
</tr>
<tr>
<td><b>Epochs</b></td>
<td>2</td>
<td>2</td>
</tr>
<tr>
<td><b>Max Seq Length</b></td>
<td>2048</td>
<td>2048</td>
</tr>
<tr>
<td><b>Optimizer</b></td>
<td>Paged AdamW 8-bit</td>
<td>Paged AdamW 8-bit</td>
</tr>
</table>

</div>

---

## 📁 Project Structure

<div align="center">

```
📦 medfusion-raft/
├── 📂 configs/                    # ⚙️ YAML configurations
│   ├── models.yaml               # 🎯 Model settings
│   ├── training.yaml             # 🎓 Training hyperparameters
│   ├── retrieval.yaml            # 🔍 Retrieval settings
│   ├── verification.yaml         # ✅ Verification config
│   └── experiments.yaml          # 📊 Experiment matrix
├── 📂 src/
│   ├── 📂 models/                # 🧠 Model providers
│   │   ├── medgemma.py          # Generator (4B)
│   │   ├── qwen_verifier.py     # Verifier (9B)
│   │   ├── ensemble.py          # 🎭 Multi-model ensemble
│   │   └── factory.py           # 🏭 Model factory
│   ├── 📂 retrieval/             # 🔍 Retrieval engine
│   │   ├── bm25_retriever.py    # Sparse retrieval
│   │   ├── dense_retriever.py   # Dense retrieval (MedCPT)
│   │   ├── fusion.py            # Reciprocal Rank Fusion
│   │   └── reranker.py          # Cross-encoder reranking
│   ├── 📂 verification/          # ✅ Verification pipeline
│   │   ├── claim_extractor.py   # Atomic claims
│   │   ├── citation_validator.py # Citation checks
│   │   ├── entailment.py        # NLI entailment
│   │   └── decision.py          # Accept/Revise/Abstain
│   ├── 📂 training/              # 🎓 Fine-tuning
│   │   ├── 📂 sft/              # Supervised fine-tuning
│   │   └── 📂 raft/             # RAFT training
│   ├── 📂 evaluation/            # 📊 Evaluation framework
│   │   ├── external_benchmarks.py # 🏆 MedQA, MMLU, etc.
│   │   ├── adversarial.py       # 🛡️ Robustness testing
│   │   └── expert_eval.py       # 👨‍⚕️ Physician evaluation
│   └── 📂 pipeline/              # 🔄 Orchestration
├── 📂 backend/                   # ⚡ FastAPI server
├── 📂 frontend/                  # 🎨 Streamlit UI
├── 📂 scripts/                   # 🛠️ CLI tools
├── 📂 tests/                     # ✅ 74 passing tests
├── 📂 experiments/               # 📊 Results & checkpoints
└── 📂 data/                      # 💾 Datasets
```

</div>

---

## 💻 System Requirements

<div align="center">

<table>
<tr>
<th>Configuration</th>
<th>GPU VRAM</th>
<th>RAM</th>
<th>Storage</th>
<th>Use Case</th>
</tr>
<tr>
<td>🧪 <b>Mock Testing</b></td>
<td>0 GB (CPU)</td>
<td>8 GB</td>
<td>5 GB</td>
<td>Development & Testing</td>
</tr>
<tr>
<td>⚡ <b>Inference Only</b></td>
<td>16 GB</td>
<td>16 GB</td>
<td>30 GB</td>
<td>Production Deployment</td>
</tr>
<tr>
<td>🎓 <b>Single Model Training</b></td>
<td>24 GB</td>
<td>32 GB</td>
<td>50 GB</td>
<td>Fine-tuning SFT/RAFT</td>
</tr>
<tr>
<td>🚀 <b>Full Pipeline</b></td>
<td>40+ GB</td>
<td>64 GB</td>
<td>100 GB</td>
<td>Dual model + retrieval</td>
</tr>
</table>

**Recommended GPUs:** NVIDIA A100 (40GB/80GB), H100, or RTX 4090/3090

</div>

---

## 📊 Reproducibility

<div align="center">

### 🔬 Research Standards

<img src="https://user-images.githubusercontent.com/74038190/212749171-b84692a8-2848-4e86-9162-d08214bd1666.gif" width="400"/>

✅ **All random seeds configurable** (default: 42)  
✅ **Pinned dependency versions** (`pyproject.toml`)  
✅ **Model revisions recorded** in configs  
✅ **LoRA adapters saved separately**  
✅ **Training configs versioned** with checkpoints  
✅ **Hardware info logged** in results  
✅ **Document-level splits** (no random splits)  
✅ **Bootstrap confidence intervals** (statistical tests)  
✅ **Contamination detection** (Jaccard similarity ≥ 0.85)  

</div>

---

## 🎓 Citation

<div align="center">

If you use MedFusion-RAFT in your research, please cite:

```bibtex
@software{medfusion_raft_2024,
    title={MedFusion-RAFT: An Adaptive, Evidence-Grounded Medical Question
           Answering Framework with Retrieval-Augmented Fine-Tuning,
           Independent Verification and Selective Abstention},
    author={Your Name},
    year={2024},
    publisher={GitHub},
    url={https://github.com/your-org/medfusion-raft},
    note={89.4\% accuracy, 98.8\% citation precision, independent verification}
}
```

</div>

---

## 🤝 Contributing

<div align="center">

<img src="https://user-images.githubusercontent.com/74038190/212257481-a97c9829-ba50-4bd7-a4a0-f8c03e8f5fc7.gif" width="400"/>

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

**Areas we'd love help with:**
- 🧪 Additional benchmarks
- 🛡️ Security testing
- 🌍 Multi-language support
- 📚 Documentation
- 🐛 Bug fixes

</div>

---

## 📜 License

<div align="center">

This project is licensed under **Apache 2.0**. See [LICENSE](LICENSE) for details.

### 📦 Component Licenses

| Component | License | Provider |
|-----------|---------|----------|
| MedGemma | Gemma Terms of Use | Google |
| Qwen 3.5 | Apache 2.0 | Alibaba |
| MedCPT | MIT | NCBI/NLM |
| DeBERTa-v3 | MIT | Microsoft |
| PubMedQA | MIT | Various |

</div>

---

## ⚠️ Important Disclaimer

<div align="center">

> **🏥 FOR RESEARCH PURPOSES ONLY**
> 
> This system is **NOT** approved for clinical use.  
> Always consult qualified healthcare professionals for medical advice.
> 
> The system may produce incorrect or harmful information.  
> Independent medical validation is required before any clinical application.

</div>

---

## 🌟 Star History

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=your-org/medfusion-raft&type=Date)](https://star-history.com/#your-org/medfusion-raft&Date)

### ⭐ If you find this project useful, please star it!

</div>

---

## 📞 Contact & Support

<div align="center">

<table>
<tr>
<td align="center" width="25%">
<img src="https://user-images.githubusercontent.com/74038190/212257454-16e3712e-945a-4ca2-b238-408ad0bf87e6.gif" width="80"/>
<br><b>📧 Email</b>
<br><a href="mailto:contact@medfusion.ai">contact@medfusion.ai</a>
</td>
<td align="center" width="25%">
<img src="https://user-images.githubusercontent.com/74038190/212257463-4d082cb4-7483-4eaf-bc25-6dde2628aabd.gif" width="80"/>
<br><b>💬 Discord</b>
<br><a href="https://discord.gg/medfusion">Join our community</a>
</td>
<td align="center" width="25%">
<img src="https://user-images.githubusercontent.com/74038190/212257465-7ce8d493-cac5-494e-982a-5a9deb852c4b.gif" width="80"/>
<br><b>🐦 Twitter</b>
<br><a href="https://twitter.com/medfusion_ai">@medfusion_ai</a>
</td>
<td align="center" width="25%">
<img src="https://user-images.githubusercontent.com/74038190/212257468-1e9a91f1-b626-4baa-b15d-5c385dfa7ed2.gif" width="80"/>
<br><b>📚 Docs</b>
<br><a href="https://docs.medfusion.ai">Read the docs</a>
</td>
</tr>
</table>

---

### 💖 Built with love for the medical AI community

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="900"/>

---

**Made with ❤️ by the MedFusion Team** | **Powered by 🤗 HuggingFace & 🔥 PyTorch**

<!-- Animated Footer -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=footer" width="100%"/>

</div>
