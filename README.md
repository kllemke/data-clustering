# Cluster-to-Purpose – Tickets #8 und #9

| Ticket | Datei | Inhalt |
| --- | --- | --- |
| #9 CSV-Upload | `cluster_to_purpose/csv_upload.py` | CSV einlesen, Trennzeichen, Encoding, Kopfzeile und Dezimalkomma erkennen, klare Fehlermeldungen |
| #8 Wissensbasis prüfen | `cluster_to_purpose/kb_validation.py` | Struktur-, Fach- und Konsistenzprüfung der Regeln, Testfälle, Review-Bericht |
| #8 Regeln | `data/knowledge_base.json` | 16 Empfehlungen und 3 Warnregeln |
| #8 Testfälle | `data/kb_test_cases.json` | 9 Datenprofile mit erwarteter Empfehlung |
| gemeinsam | `cluster_to_purpose/knowledge_base.py` | Wissensbasis laden, Regeln gegen ein Datenprofil abgleichen |

## Einrichten (einmalig)

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Mac / Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## Benutzen

```bash
# alle Tests
python -m pytest

# eine CSV prüfen
python -m cluster_to_purpose.csv_upload pfad/zur/datei.csv

# Wissensbasis prüfen und Review-Bericht schreiben
python -m cluster_to_purpose.kb_validation --report kb_review.md
```

## CSV-Upload im eigenen Code

```python
from cluster_to_purpose.csv_upload import load_csv, CSVUploadError

try:
    result = load_csv("daten.csv")        # oder Bytes oder ein Datei-Objekt
except CSVUploadError as err:
    print(err)                            # Meldung direkt für den Nutzer
else:
    df = result.df                        # geht weiter an das Profiling (#14)
    print(result.summary(), result.warnings)
```

Mit Streamlit (falls das Team das nutzt):

```python
import streamlit as st
from cluster_to_purpose.csv_upload import load_csv, CSVUploadError

file = st.file_uploader("CSV hochladen", type=["csv", "txt", "tsv"])
if file:
    try:
        result = load_csv(file)
        st.dataframe(result.preview(10))
        for w in result.warnings:
            st.warning(w)
    except CSVUploadError as err:
        st.error(str(err))
```

## Datenprofil: Absprache mit #14, #18, #19 und #22

Die Wissensbasis erwartet ein Profil in diesem Format:

```python
{
    "data_type": "numeric",   # numeric | categorical | mixed | time_series | text | stream | uncertain
    "n_rows": 150,            # Anzahl Objekte (Zeilen bzw. Zeitreihen)
    "n_features": 4,          # Anzahl Merkmale bzw. Zeitreihenlänge
    "outlier_ratio": 0.01,    # Anteil Ausreißer 0..1, aus #19
}
```

`match_rules(profile, rules)` liefert `status` (`ok` oder `uncertain`), passende Empfehlungen und Warnungen.
Die Empfehlungs-Engine (#22) kann darauf aufbauen und das Ranking ergänzen.

## Noch offen für das Review (#8)

- Für jede Regel `paper_section` eintragen: In welchem Abschnitt des Kurs-Papers steht die Aussage?
  Solange das fehlt, meldet der Validator eine Warnung.
- Regeln mit dem Team abgleichen, falls #23 die Wissensbasis anders aufbaut.
