from __future__ import annotations


DISCLAIMER = "This system is for orientation only and does not replace medical consultation."


def build_recommendation(urgency: str, specialty: str | None = None) -> str:
    specialty_text = f" This message may correspond to symptoms usually handled by {specialty}." if specialty else ""
    if urgency == "high":
        return (
            "This may require urgent medical attention. Please contact emergency services or visit the nearest "
            "healthcare facility."
            f"{specialty_text}"
        )
    if urgency == "medium":
        return f"Please seek timely advice from a qualified healthcare professional if symptoms persist or worsen.{specialty_text}"
    if urgency == "low":
        return f"This appears suitable for a non-urgent medical consultation if symptoms persist.{specialty_text}"
    return f"Please consult a qualified healthcare professional for orientation.{specialty_text}"
