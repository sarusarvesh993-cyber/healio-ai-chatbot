import io
import os
from datetime import datetime
from typing import Dict, List, Optional
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

class DoctorSummaryGenerator:
    def generate_markdown(
        self,
        patient_name: str = "Anonymous Patient",
        age_gender: str = "Not specified",
        chief_complaint: str = "",
        symptoms_history: str = "",
        medications: str = "",
        allergies: str = "None reported",
        vitals: str = "Not recorded",
        questions: str = "",
        conversation_summary: str = ""
    ) -> str:
        date_str = datetime.now().strftime("%B %d, %Y - %I:%M %p")
        return f"""# 📋 CLINICAL ENCOUNTER SUMMARY FOR PHYSICIAN CONSULTATION
**Document Generated:** {date_str} | **System:** MediCare AI Clinical Companion

---

### 👤 Patient Information
- **Patient Name:** {patient_name}
- **Age / Gender:** {age_gender}
- **Current Reported Vitals:** {vitals if vitals else 'None recorded'}

---

### 🩺 Chief Complaint & History of Present Illness (HPI)
- **Primary Concern:** {chief_complaint if chief_complaint else 'General healthcare query'}
- **Timeline & Severity:** {symptoms_history if symptoms_history else 'Discussed during AI consultation'}

---

### 💊 Active Medications & Allergies
- **Current Medications:** {medications if medications else 'None reported'}
- **Known Drug / Food Allergies:** {allergies if allergies else 'None reported'}

---

### 💬 Summary of Patient Concerns & AI Triage Context
{conversation_summary if conversation_summary else 'Patient engaged in conversational symptom review.'}

---

### ❓ Prepared Questions for the Physician
{questions if questions else '1. Are further diagnostic lab tests indicated for these symptoms?\n2. Are any current medications interacting or causing side effects?\n3. What warning signs should prompt immediate medical follow-up?'}

---
*Confidential Medical Information — Prepared for Doctor-Patient Consultation Review.*
"""

    def generate_pdf_bytes(
        self,
        patient_name: str = "Anonymous Patient",
        age_gender: str = "Not specified",
        chief_complaint: str = "",
        symptoms_history: str = "",
        medications: str = "",
        allergies: str = "None reported",
        vitals: str = "Not recorded",
        questions: str = "",
        conversation_summary: str = ""
    ) -> bytes:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#0077B6'),
            spaceAfter=6
        )
        h2_style = ParagraphStyle(
            'H2Style',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#023E8A'),
            spaceBefore=10,
            spaceAfter=4
        )
        body_style = ParagraphStyle(
            'BodyStyle',
            parent=styles['Normal'],
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor('#1E293B')
        )

        elements = []

        elements.append(Paragraph("<b>MediCare AI — Clinical Encounter Summary</b>", title_style))
        date_str = datetime.now().strftime("%B %d, %Y - %I:%M %p")
        elements.append(Paragraph(f"<i>Generated on {date_str} for Physician Consultation</i>", body_style))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0077B6'), spaceAfter=10))

        patient_data = [
            [Paragraph("<b>Patient Name:</b>", body_style), Paragraph(patient_name, body_style),
             Paragraph("<b>Age / Gender:</b>", body_style), Paragraph(age_gender, body_style)],
            [Paragraph("<b>Allergies:</b>", body_style), Paragraph(allergies, body_style),
             Paragraph("<b>Vitals:</b>", body_style), Paragraph(vitals, body_style)],
        ]
        t = Table(patient_data, colWidths=[1.1*inch, 2.3*inch, 1.1*inch, 2.3*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F0F9FF')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#BAE6FD')),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 10))

        elements.append(Paragraph("<b>Chief Complaint & Symptom Timeline</b>", h2_style))
        elements.append(Paragraph(f"<b>Primary Concern:</b> {chief_complaint}", body_style))
        if symptoms_history:
            elements.append(Paragraph(f"<b>History / Duration:</b> {symptoms_history}", body_style))
        elements.append(Spacer(1, 8))

        elements.append(Paragraph("<b>Active Medications</b>", h2_style))
        elements.append(Paragraph(medications if medications else "None reported by patient.", body_style))
        elements.append(Spacer(1, 8))

        if conversation_summary:
            elements.append(Paragraph("<b>Clinical Consultation & Triage Notes</b>", h2_style))
            elements.append(Paragraph(conversation_summary.replace("\n", "<br/>"), body_style))
            elements.append(Spacer(1, 8))

        elements.append(Paragraph("<b>Recommended Questions for the Healthcare Provider</b>", h2_style))
        if questions:
            for q in questions.split("\n"):
                if q.strip():
                    elements.append(Paragraph(f"• {q.strip()}", body_style))
        else:
            elements.append(Paragraph("• What diagnostic evaluations or labs are recommended?", body_style))
            elements.append(Paragraph("• Do my current medications need any dosage adjustment?", body_style))

        elements.append(Spacer(1, 14))
        elements.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#94A3B8'), spaceAfter=6))
        elements.append(Paragraph("<font size=7.5 color='#64748B'><b>CONFIDENTIALITY NOTICE:</b> This summary was generated by MediCare AI as an educational patient companion to facilitate clinical communication with a licensed medical professional.</font>", body_style))

        doc.build(elements)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes
