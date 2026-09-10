# 🏥 HEALIO — Advanced Clinical Intelligence & Public Health Platform

[![Vercel Deployment](https://img.shields.io/badge/Vercel-Live_Deployment-black?logo=vercel)](https://healio-ai-chatbot-3gloycz0q-sarusarvesh993-cybers-projects.vercel.app)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.110-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v3.4-38B2AC?logo=tailwindcss&logoColor=white)](https://tailwindcss.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**HEALIO** is an enterprise-grade, multi-modal clinical intelligence platform designed to democratize public healthcare price transparency, automate insurance claim denial appeals, provide real-time evidence-based clinical guidance, evaluate drug interactions, and assess emergency clinical acuity.

---

## 🌟 Core Clinical Modules

| Module | Description | Key Capabilities |
| :--- | :--- | :--- |
| 💰 **Generic Price-Saver** | Public Medicine Price Transparency | Identifies bioequivalent salt compounds & PMBJP codes, calculating **70%–90% price savings** vs. commercial brands. |
| 📑 **Insurance Appeal Generator** | Legal Dispute Letter Drafter | Decodes unfair hospital bill deductions (consumables, room-rent capping, PED clauses) and compiles downloadable IRDAI-compliant dispute PDFs. |
| 💬 **AI Clinical Companion** | Multi-Provider Medical Chat | Supports Groq (`llama-3.1-8b-instant`), OpenRouter, and OpenAI models with multi-language translation and 4-7-8 calming respiratory protocol. |
| 🔬 **Vision Lab Scanner** | Multi-Modal Parameter Extraction | Extracts clinical biomarker panels (Blood Glucose, HbA1c, Serum Creatinine, Lipids) with color-coded gauge status bars. |
| 💊 **OpenFDA Drug Matrix** | Pharmacological Safety Engine | Checks pharmacokinetic contraindications, drug-drug synergy risks, food/alcohol interactions, and dosage safety. |
| 🩺 **ESI Clinical Triage** | Emergency Acuity Assessment | Multi-step guided symptom triage scoring clinical severity from **Level 1 (Resuscitation)** to **Level 5 (Non-Urgent)**. |

---

## 🏗️ Technical Architecture
┌────────────────────────────────────────────────────────────────────────┐
│ HEALIO CLINICAL PLATFORM │
├────────────────────────────────┬───────────────────────────────────────┤
│ 🌐 Modern Single-Page App │ Tailwind CSS, jsPDF, Marked.js, Web-API│
│ ⚡ Fast Backend Microservice │ FastAPI, Uvicorn, Pydantic, CORS │
│ 🤖 LLM Multi-Provider Engine │ Groq, OpenRouter, OpenAI │
│ 📊 Clinical Knowledge Base │ PMBJP Formulary, OpenFDA Safety Matrix │
│ 🚀 Serverless Deployment │ Vercel Python Zero-Config Runtime │
└────────────────────────────────┴───────────────────────────────────────┘

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/sarusarvesh993-cyber/healio-ai-chatbot.git
cd healio-ai-chatbot

# Install dependencies
pip install -r requirements.txt
