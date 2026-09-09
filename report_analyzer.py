import base64
import io
import os
import json
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import requests
import pypdf
from api_client import UniversalLLMClient

REPORT_ANALYSIS_PROMPT = """You are MediCare AI's Senior Clinical Diagnostic & Pharmacological Assistant.
Your mission is to analyze the provided medical lab report, diagnostic test, or doctor's prescription with high clinical accuracy, empathy, and patient-friendly clarity.
Respond fluently in {language}.

Analyze the image or extracted document and provide a structured, beautifully formatted report in Markdown.

Use the following exact structure:

# 📋 Medical Report & Prescription Analysis

### 📌 Summary Overview
- **Document Type:** (e.g., Doctor Prescription, Lipid Profile, Complete Blood Count (CBC), Thyroid Panel)
- **Detected Date / Doctor / Lab:** (if visible, otherwise "Not specified")
- **Overall Assessment:** (1-2 sentences summarizing the findings or prescribed treatment in warm, clear language)

---

### 💊 Prescribed Medications & Dosages (If Prescription)
| Medication Name | Generic / Category | Strength & Dosage | Frequency & Duration | Primary Purpose |
| :--- | :--- | :--- | :--- | :--- |
| (e.g. Orlistat) | Lipase Inhibitor | 120 mg | 1 cap PO, 3 times daily | Weight management / lipid reduction |
| (e.g. Metformin) | Biguanide | 500-1000 mg | Once daily | Blood glucose regulation |

---

### 📊 Test Parameters Breakdown (If Lab Report)
| Biomarker / Test | Your Value | Standard Range | Unit | Status |
| :--- | :--- | :--- | :--- | :--- |
| (e.g. Fasting Blood Glucose) | 135 | 70 - 99 | mg/dL | 🔴 High |
| (e.g. HDL Cholesterol) | 52 | > 40 | mg/dL | 🟢 Normal |
| (e.g. Total Cholesterol) | 205 | < 200 | mg/dL | 🟡 Borderline |

*Use these status badges where applicable: 🟢 Normal, 🟡 Borderline, 🔴 High / Low.*

---

### 💡 Plain-English Clinical Explanations & Advice
For every key medication or abnormal value:
- **How it works:** Explain what the medicine or biomarker does in simple, jargon-free words.
- **Key Precautions:** Important instructions (e.g., take with meals, avoid skipping doses, avoid certain foods).
- **Potential Side Effects:** Common things to monitor.

---

### 🥗 Lifestyle, Nutrition & Patient Guidance
Provide 3-5 practical, evidence-based lifestyle habits mentioned or recommended for this case (e.g., diet, aerobic exercise, portion control, hydration, sleep).

---

### 🩺 Smart Questions for Your Doctor / Pharmacist
Provide 3 to 5 targeted questions the patient should ask:
1. "..."
2. "..."
3. "..."

---

### ⚠️ Medical Disclaimer
*This AI analysis is provided for educational and informational purposes only and does not constitute formal medical diagnosis, treatment, or clinical consultation. Always consult your prescribing physician or pharmacist.*
"""

class ReportAnalyzer:
    def __init__(self, api_key: Optional[str] = None, provider: str = "auto", model_name: Optional[str] = None):
        self.api_key = (api_key or os.getenv("OPENROUTER_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")).strip()
        self.provider = provider
        self.model_name = (model_name or "openai/gpt-4o-mini").strip()
        self.llm_client = UniversalLLMClient(api_key=self.api_key, provider=self.provider, model=self.model_name)

    def update_config(self, api_key: Optional[str] = None, provider: str = "auto", model_name: Optional[str] = None):
        if api_key:
            self.api_key = api_key.strip()
        self.provider = provider
        if model_name:
            self.model_name = model_name.strip()
        self.llm_client = UniversalLLMClient(api_key=self.api_key, provider=self.provider, model=self.model_name)

    def _encode_image(self, image_file) -> Tuple[str, str]:
        if isinstance(image_file, bytes):
            image = Image.open(io.BytesIO(image_file))
        elif hasattr(image_file, "read"):
            image = Image.open(image_file)
        else:
            image = image_file

        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

        max_dim = 1600
        if max(image.size) > max_dim:
            image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return base64_str, "image/jpeg"

    def _extract_pdf_text_and_images(self, pdf_file) -> Tuple[str, List[Image.Image]]:
        text_content = ""
        extracted_images = []

        if hasattr(pdf_file, "read"):
            pdf_bytes = io.BytesIO(pdf_file.read())
        else:
            pdf_bytes = io.BytesIO(pdf_file)

        try:
            reader = pypdf.PdfReader(pdf_bytes)
            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_content += f"\n--- Page {page_idx + 1} ---\n" + page_text

                for img_obj in page.images:
                    try:
                        img = Image.open(io.BytesIO(img_obj.data))
                        extracted_images.append(img)
                    except Exception:
                        pass
        except Exception as e:
            text_content = f"Error reading PDF text: {str(e)}"

        return text_content.strip(), extracted_images

    def analyze(self, file_data, file_type: str, custom_notes: str = "", api_key: Optional[str] = None, provider: str = "auto", model_name: Optional[str] = None, language: str = "English") -> Dict:
        active_key = (api_key or self.api_key or os.getenv("OPENROUTER_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")).strip()

        if not active_key:
            return {
                "success": False,
                "error": "API Key not found. Please paste your API key in the sidebar and click 'Apply Key'.",
            }

        user_prompt = REPORT_ANALYSIS_PROMPT.format(language=language)
        if custom_notes.strip():
            user_prompt += f"\n\n**Patient's Additional Context / Symptoms:**\n{custom_notes.strip()}"

        try:
            if "pdf" in file_type.lower():
                pdf_text, pdf_images = self._extract_pdf_text_and_images(file_data)
                if pdf_images and len(pdf_images) > 0:
                    base64_img, mime_type = self._encode_image(pdf_images[0])
                    prompt_text = user_prompt + (f"\n\nExtracted PDF Text:\n{pdf_text}" if pdf_text else "")
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime_type};base64,{base64_img}"},
                                },
                            ],
                        }
                    ]
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": f"{user_prompt}\n\n**DOCUMENT CONTENT (Extracted from PDF):**\n{pdf_text}",
                        }
                    ]
            else:
                base64_img, mime_type = self._encode_image(file_data)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{base64_img}"},
                            },
                        ],
                    }
                ]

            url, headers, default_model = self.llm_client.get_endpoint_and_headers(active_key, provider)
            target_model = (model_name or self.model_name or default_model).replace(":free", "")

            candidate_models = [target_model, "openai/gpt-4o-mini", "gpt-4o-mini", "google/gemini-flash-1.5", "meta-llama/llama-3.2-11b-vision-instruct:free", "openrouter/auto"]

            last_error = ""
            for mod_id in list(dict.fromkeys(candidate_models)):
                if not mod_id:
                    continue
                payload = {
                    "model": mod_id,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 2500,
                }

                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=100)
                    if response.status_code == 200:
                        res_data = response.json()
                        result_text = res_data["choices"][0]["message"]["content"]
                        return {
                            "success": True,
                            "analysis": result_text,
                            "model_used": res_data.get("model", mod_id),
                        }
                    else:
                        try:
                            last_error = response.json().get("error", {}).get("message", response.text)
                        except Exception:
                            last_error = response.text
                except Exception as req_err:
                    last_error = str(req_err)

            return {
                "success": False,
                "error": f"API Error: {last_error}",
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to analyze report: {str(e)}",
            }
