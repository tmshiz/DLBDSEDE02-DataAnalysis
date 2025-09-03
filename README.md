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
  --out_txt<br/>                definiert den Speicherort der Vorverarbeitungsphase - Standard: c:\Users\username\Desktop<br/>
  --out_png<br/>                definiert den Speicherort der Themenübersicht (PNG) - Standard: c:\Users\username\Desktop<br/>
  --out_top<br/>                definiert den Speicherort der Themenübersicht (TXT) - Standard: c:\Users\username\Desktop<br/>
  --topics 10<br/>               Anzahl der Themen - Standard: 10<br/>
  --label_words 3<br/>        Anzahl der Wörter pro Thema im Label - Standard: 3<br/>
  --top_words 10<br/>           Anzahl der definierten Top-Wörter pro Thema  - Standard: 10<br/>
  --spellcheck<br/>             aktiviert die Rechtschreibung  - Standard: deaktiviert<br/>
<br/>
<p/>
Dateien:<br/>
Requirements.txt           (enthält die erforderlichen Bibliotheken und deren Version)<br/>
meldungen.json             (Datensatz mit 32813 Einträgen)<br/>
meldungen_1000.json        (Datensatz mit 1000 Einträgen)
