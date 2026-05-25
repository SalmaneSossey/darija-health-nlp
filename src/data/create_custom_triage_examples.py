from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, ensure_project_dirs


Scenario = tuple[str, str, str, str, str]

BASE_EXAMPLES: list[Scenario] = [
    ("3ndi wje3 f sedri w di9 f nefs", "latin_darija", "Cardiology", "high", "chest_pain;shortness_of_breath"),
    ("wja3 f sedri m3a nefs m9t3", "latin_darija", "Cardiology", "high", "chest_pain;shortness_of_breath"),
    ("kan7ess b mal a la poitrine w di9 f nefs", "mixed", "Cardiology", "high", "chest_pain;shortness_of_breath"),
    ("عندي ألم فالصدر وضيق فالتنفس", "arabic_darija", "Cardiology", "high", "chest_pain;shortness_of_breath"),
    ("mal a la poitrine avec difficulte a respirer", "french", "Cardiology", "high", "chest_pain;shortness_of_breath"),
    ("douleur thoracique et souffle court", "french", "Cardiology", "high", "chest_pain;shortness_of_breath"),
    ("3ndi di9 f nefs fach kan tla3 droj", "latin_darija", "Pulmonology", "high", "shortness_of_breath"),
    ("عندي ضيق في التنفس وكحة", "arabic_darija", "Pulmonology", "high", "shortness_of_breath;cough"),
    ("j ai du mal a respirer avec toux", "french", "Pulmonology", "high", "shortness_of_breath;cough"),
    ("waldi 3ndo skhana w k7a", "latin_darija", "Pediatric Medicine", "medium", "fever;cough"),
    ("benti 3andha skhana w toux", "mixed", "Pediatric Medicine", "medium", "fever;cough"),
    ("طفلي عندو سخانة وكحة", "arabic_darija", "Pediatric Medicine", "medium", "fever;cough"),
    ("mon enfant a de la fievre et une toux", "french", "Pediatric Medicine", "medium", "fever;cough"),
    ("j ai une toux et de la fievre", "french", "General Practice", "medium", "cough;fever"),
    ("عندي كحة وسخانة ثلاثة أيام", "arabic_darija", "General Practice", "medium", "cough;fever"),
    ("3ndi skhana mn 3 iyam w k7a", "latin_darija", "General Practice", "medium", "fever;cough"),
    ("3ndi sda3 w dwakha", "latin_darija", "Neurology", "low", "headache;dizziness"),
    ("صداع خفيف ودوخة", "arabic_darija", "Neurology", "low", "headache;dizziness"),
    ("j ai mal a la tete et vertige", "french", "Neurology", "low", "headache;dizziness"),
    ("sda3 qwi w kaydour lia rassi", "latin_darija", "Neurology", "medium", "headache;dizziness"),
    ("3ndi haboub f wejhi", "latin_darija", "Dermatology", "low", "skin_rash"),
    ("3ndi 7boub sghar f wejhi", "latin_darija", "Dermatology", "low", "skin_rash"),
    ("عندي طفح جلدي وحكة", "arabic_darija", "Dermatology", "low", "skin_rash"),
    ("boutons sur le visage avec demangeaison", "french", "Dermatology", "low", "skin_rash"),
    ("haboub f wejhi w kay7kouni", "latin_darija", "Dermatology", "low", "skin_rash"),
    ("j ai mal au ventre depuis 3 jours", "french", "Gastroenterology", "medium", "stomach_pain"),
    ("3ndi wje3 f kerchi mn lbareh", "latin_darija", "Gastroenterology", "medium", "stomach_pain"),
    ("kan t9ya bzaf w kerchi katwje3ni", "latin_darija", "Gastroenterology", "medium", "vomiting;stomach_pain"),
    ("عندي وجع فكرشي وترجيع", "arabic_darija", "Gastroenterology", "medium", "stomach_pain;vomiting"),
    ("mal au ventre avec vomissements", "french", "Gastroenterology", "medium", "stomach_pain;vomiting"),
    ("nziif qwi w ma7bssch", "latin_darija", "Emergency Medicine", "high", "bleeding"),
    ("نزيف قوي ومحبسش", "arabic_darija", "Emergency Medicine", "high", "bleeding"),
    ("saignement fort qui ne s arrete pas", "french", "Emergency Medicine", "high", "bleeding"),
    ("ghmi 3lia w t7t lard", "latin_darija", "Emergency Medicine", "high", "loss_of_consciousness"),
    ("غمي عليا وفقدت الوعي", "arabic_darija", "Emergency Medicine", "high", "loss_of_consciousness"),
    ("perte de conscience soudaine", "french", "Emergency Medicine", "high", "loss_of_consciousness"),
    ("7amla w kayn nziif", "latin_darija", "Obstetrics and Gynecology", "high", "pregnancy_bleeding"),
    ("حاملة وعندي نزيف", "arabic_darija", "Obstetrics and Gynecology", "high", "pregnancy_bleeding"),
    ("saignement pendant la grossesse", "french", "Obstetrics and Gynecology", "high", "pregnancy_bleeding"),
    ("rassi kaywje3ni chwiya", "latin_darija", "General Practice", "low", "headache"),
    ("عندي سخانة خفيفة وعياء", "arabic_darija", "General Practice", "low", "fever"),
    ("petite fievre et fatigue", "french", "General Practice", "low", "fever"),
]

CONTEXT_SUFFIXES: dict[str, list[str]] = {
    "latin_darija": [
        "",
        " mn lbareh",
        " lyouma",
        " mn 3 iyam",
        " w kan7ess b l3ya",
        " wach khasni nmchi ltbib",
        " w ma3reftch chno ndir",
        " bla 7rara kbira",
        " kayji w kaymchi",
        " w kayzid f lil",
        " f sbah kayn bzzaf",
        " m3a chwiya dyal lqlaq",
    ],
    "arabic_darija": [
        "",
        " من البارح",
        " اليوم",
        " من ثلاثة أيام",
        " وكنحس بالتعب",
        " واش خاصني نمشي لطبيب",
        " وماعرفتش شنو ندير",
        " بلا سخانة كبيرة",
        " كيجي وكي مشي",
        " وكيزيد فالليل",
        " فالصباح كاين بزاف",
        " مع شوية ديال القلق",
    ],
    "french": [
        "",
        " depuis hier",
        " aujourd hui",
        " depuis trois jours",
        " avec fatigue",
        " est ce que je dois consulter",
        " je ne sais pas quoi faire",
        " sans forte fievre",
        " ca vient et ca part",
        " plus fort la nuit",
        " surtout le matin",
        " avec un peu d anxiete",
    ],
    "mixed": [
        "",
        " mn lbareh",
        " aujourd hui",
        " depuis 3 jours",
        " w kan7ess fatigue",
        " wach consulter daba",
        " ma3reftch quoi faire",
        " bla forte fievre",
        " kayji w kaymchi",
        " plus fort la nuit",
        " surtout f sbah",
        " m3a un peu d anxiete",
    ],
}


def create_custom_examples() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for text, language, specialty, urgency, symptoms in BASE_EXAMPLES:
        for suffix in CONTEXT_SUFFIXES[language]:
            rows.append(
                {
                    "text": f"{text}{suffix}".strip(),
                    "language": language,
                    "specialty": specialty,
                    "urgency": urgency,
                    "symptoms": symptoms,
                    "source": "custom",
                }
            )
    df = pd.DataFrame(rows).drop_duplicates(subset=["text"])
    return df.sort_values(["language", "specialty", "text"]).reset_index(drop=True)


def main() -> None:
    ensure_project_dirs()
    df = create_custom_examples()
    output_path = PROCESSED_DATA_DIR / "custom_triage_examples.csv"
    df.to_csv(output_path, index=False)
    df.head(40).to_csv(SAMPLE_DATA_DIR / "sample_custom_triage_examples.csv", index=False)
    print(f"Saved {len(df)} custom triage examples to {output_path}")


if __name__ == "__main__":
    main()
