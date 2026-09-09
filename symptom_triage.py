import os
from typing import Dict, List, Optional
from api_client import UniversalLLMClient

class SymptomTriageEngine:
    def __init__(self, llm_client: Optional[UniversalLLMClient] = None):
        self.llm_client = llm_client or UniversalLLMClient()

    def evaluate_triage(
        self,
        body_part: str,
        primary_symptom: str,
        pain_level: int,
        duration: str,
        associated_symptoms: List[str],
        patient_age_group: str = "Adult (18-64)",
        api_key: str = "",
        provider: str = "auto",
        model: str = "",
        language: str = "English"
    ) -> str:
        is_red_flag = pain_level >= 9 or any(
            sym in ["Chest pain radiating to arm/jaw", "Sudden weakness / numbness on one side", "Severe difficulty breathing", "Loss of consciousness / fainting", "Sudden severe thunderclap headache"]
            for sym in associated_symptoms
        ) or ("Chest" in body_part and pain_level >= 7)

        prompt = f"""You are MediCare AI's Senior Emergency Triage Officer & Clinical Symptom Assessor.
Respond fluently in {language}.

Evaluate the patient's symptoms using the Emergency Severity Index (ESI) Framework:

PATIENT PROFILE & TRIAGE INPUTS:
- **Anatomical Region:** {body_part}
- **Primary Complaint:** {primary_symptom}
- **Pain Severity (1-10 Scale):** {pain_level}/10
- **Onset & Duration:** {duration}
- **Age Demographic:** {patient_age_group}
- **Associated Symptoms:** {', '.join(associated_symptoms) if associated_symptoms else 'None reported'}

Format your assessment in clean, professional Markdown with the following structure:

# 🩺 Clinical Triage & Acuity Assessment

### 🚨 ESI Acuity Score & Triage Level
- **Acuity Rating:** {'🔴 LEVEL 1/2: IMMEDIATE EMERGENCY' if is_red_flag else '🟡 LEVEL 3: URGENT CARE (Within 2-4 Hours)' if pain_level >= 5 else '🟢 LEVEL 4/5: NON-URGENT / ROUTINE CARE'}
- **Recommended Care Setting:** (e.g. Emergency Department (ED), Urgent Care Clinic, Primary Care Physician, or Home Monitoring)
- **Timeframe for Medical Evaluation:** (e.g., Immediately, Within 24 hours, Next few days)

---

### 🔍 Clinical Symptom Breakdown
- **Potential Underlying Factors:** (List 2-4 plausible common non-definitive clinical possibilities in empathetic language)
- **Risk Assessment:** Explain why this combination of symptoms (pain level {pain_level}/10, duration {duration}) warrants this specific urgency level.

---

### 🚩 Red Flag Warning Signs (When to Call Emergency Immediately)
List 3-5 critical warning signs that would require calling 911 / 108 instantly.

---

### 🩹 Immediate Comfort & First-Aid Measures
Provide 3 safe, evidence-based supportive home-care steps (e.g., resting, hydration, ice/heat protocol, elevation) that are safe before seeing a clinician.

---

### 🩺 What to Tell Your Healthcare Provider
Give the patient a structured 2-sentence summary they can read directly to the triage nurse or doctor.

---

### ⚠️ Medical Triage Disclaimer
*This automated triage assessment is an educational screening guide based on the Emergency Severity Index and does not replace in-person clinical judgment. If your condition deteriorates, seek emergency medical care immediately.*
"""

        messages = [{"role": "user", "content": prompt}]
        return self.llm_client.complete(
            messages=messages,
            api_key=api_key,
            provider=provider,
            model=model,
            temperature=0.2,
            max_tokens=2200
        )
