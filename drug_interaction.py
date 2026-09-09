import os
import json
import requests
from typing import Dict, List, Optional
from api_client import UniversalLLMClient

class DrugInteractionEngine:
    def __init__(self, llm_client: Optional[UniversalLLMClient] = None):
        self.llm_client = llm_client or UniversalLLMClient()

    def fetch_openfda_info(self, drug_name: str) -> Dict:
        clean_name = drug_name.strip()
        url = f'https://api.fda.gov/drug/label.json?search=openfda.brand_name:"{clean_name}"+openfda.generic_name:"{clean_name}"&limit=1'
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                results = data.get("results", [])
                if results:
                    first = results[0]
                    return {
                        "found": True,
                        "brand_name": first.get("openfda", {}).get("brand_name", [clean_name])[0],
                        "generic_name": first.get("openfda", {}).get("generic_name", ["Unknown"])[0],
                        "warnings": (first.get("warnings", ["None noted"])[0])[:300] + "...",
                        "drug_interactions": (first.get("drug_interactions", ["No specific interaction text found"])[0])[:400] + "...",
                        "food_interactions": (first.get("food_interactions", ["No specific food warnings listed"])[0])[:300] + "...",
                    }
        except Exception:
            pass
        return {"found": False, "drug_name": clean_name}

    def check_interactions(self, drugs: List[str], patient_conditions: str = "", api_key: str = "", provider: str = "auto", model: str = "", language: str = "English") -> str:
        if not drugs or len(drugs) == 0:
            return "Please enter at least one medication."

        fda_summaries = []
        for d in drugs:
            fda_data = self.fetch_openfda_info(d)
            if fda_data.get("found"):
                fda_summaries.append(f"FDA Label for {fda_data['brand_name']} ({fda_data['generic_name']}): Warnings: {fda_data['warnings']} | Interactions: {fda_data['drug_interactions']}")

        fda_context = "\n".join(fda_summaries)

        prompt = f"""You are MediCare AI's Senior Pharmacologist & Clinical Drug Safety Expert.
Respond fluently in {language}.

Analyze the following list of medications for:
1. Drug-to-Drug Interactions (Contraindications, synergistic adverse effects, enzyme competition)
2. Food & Beverage Interactions (e.g. Grapefruit, Alcohol, Dairy, Vitamin K foods, Caffeine)
3. Disease / Condition Contraindications (Patient conditions: {patient_conditions if patient_conditions else 'None specified'})
4. Suggested Dosage Timing & Administration Tips (e.g., Take with food, morning vs night)

MEDICATIONS TO EVALUATE:
{', '.join(drugs)}

FDA DATABASE LABEL CONTEXT:
{fda_context if fda_context else 'Standard pharmacology knowledge base'}

Format your answer in clean, beautiful Markdown with the following structure:

# 💊 Drug Interaction & Medication Safety Report

### 📋 Evaluated Medications
(List each drug with its category/class)

### ⚠️ Interaction Severity Matrix
| Medication Pair | Risk Level | Mechanism / Risk | Action Required |
| :--- | :--- | :--- | :--- |
| (Drug A + Drug B) | 🔴 Severe / 🟡 Moderate / 🟢 Mild | (Brief explanation) | (e.g., Avoid combination, space 4 hours apart) |

*(Use 🔴 Severe / Contraindicated, 🟡 Moderate / Monitor, 🟢 Minor / Safe)*

### 🥗 Critical Food & Alcohol Warnings
- **Alcohol:** ...
- **Dietary / Food Restrictions:** (e.g., grapefruit, leafy greens, dairy)

### ⏰ Optimal Schedule & Administration Protocol
Suggest a sample daily timetable to take these medications safely without overlap.

### 🩺 Recommended Questions for Doctor / Pharmacist
3-4 specific questions to verify with the physician.

### ⚠️ Pharmacology Disclaimer
*This pharmacology analysis is an educational screening tool and does not replace the expert judgment of a licensed physician or pharmacist.*
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
