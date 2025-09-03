# DLBDSEDE02-DataAnalysis

Enthalten sind zwei Python Scripte zur Analyse und Extraktion häufiger Themen aus einem unstrukturierten json Datensatz mit NLP Techniken.

nlp_bertopic.py: Vorverarbeitung + Vektorisierung und Themenextraktion mit BERTopic und SentenceTransformer
nlp_tfidf.py:    Vorverarbeitung + Vektorisierung und Themenextraktion mit TF-IDF und LDA

Aufruf:

python nlp_bertopic.py --input meldungen.json

python nlp_tfidf.py --input meldungen.json

Optionale Parameter:
  --out_txt                (definiert den Speicherort der Ergebnisse der Vorverarbeitungsphase - Standardverzeichnis c:\Users\username\Desktop)
  --out_png                (definiert den Speicherort der Themenübersicht (PNG) - Standardverzeichnis c:\Users\username\Desktop)
  --out_top                (definiert den Speicherort der Themenübersicht (TXT) - Standardverzeichnis c:\Users\username\Desktop)
  --topics 8               (Anzahl der Themen)
  --label_words 4          (Anzahl der Wörter pro Thema im Label)
  --top_words 12           (Anzahl der definierten Top-Wörter pro Thema)
  --spellcheck             (aktiviert die Rechtschreibung)

Dateien:
Requirements.txt           (enthält die erforderlichen Bibliotheken und deren Version)
meldungen.json             (Datensatz mit 32813 Einträgen)
meldungen_1000.json        (Datensatz mit 1000 Einträgen)
