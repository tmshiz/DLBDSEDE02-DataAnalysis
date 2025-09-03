# DLBDSEDE02-DataAnalysis <br/><br/>

Enthalten sind zwei Python Scripte zur Analyse und Extraktion häufiger Themen aus einem unstrukturierten json Datensatz mit NLP Techniken.<br/>
<p>
nlp_bertopic.py: Vorverarbeitung + Vektorisierung und Themenextraktion mit BERTopic und SentenceTransformer <br/>
nlp_tfidf.py:    Vorverarbeitung + Vektorisierung und Themenextraktion mit TF-IDF und LDA <br/><br/>
</p>
Aufruf: <br/>
<br/>
python nlp_bertopic.py --input meldungen.json <br/>
<br/>
python nlp_tfidf.py --input meldungen.json <br/>
<br/>
Optionale Parameter: <br/>
<p>
  --out_txt                (definiert den Speicherort der Ergebnisse der Vorverarbeitungsphase - Standardverzeichnis c:\Users\username\Desktop) <br/>
  --out_png                (definiert den Speicherort der Themenübersicht (PNG) - Standardverzeichnis c:\Users\username\Desktop)<br/>
  --out_top                (definiert den Speicherort der Themenübersicht (TXT) - Standardverzeichnis c:\Users\username\Desktop)<br/>
  --topics 8               (Anzahl der Themen)<br/>
  --label_words 4          (Anzahl der Wörter pro Thema im Label)<br/>
  --top_words 12           (Anzahl der definierten Top-Wörter pro Thema)<br/>
  --spellcheck             (aktiviert die Rechtschreibung)<br/>
<br/>
<p/>
Dateien:<br/>
Requirements.txt           (enthält die erforderlichen Bibliotheken und deren Version)<br/>
meldungen.json             (Datensatz mit 32813 Einträgen)<br/>
meldungen_1000.json        (Datensatz mit 1000 Einträgen)
