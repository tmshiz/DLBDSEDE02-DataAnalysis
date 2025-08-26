# DLBDSEDE02-DataAnalysis

Enthalten sind zwei Python Script zur Analyse und Extraktion häufiger Themen aus einem unstrukturierten json Datensatz mit NLP Techniken.

nlp_bertopic.py: Umsetzung mit BERTopic und SentenceTransformer
nlp_tfidf.py:    Umsetzung mit TF-IDF und LDA

Der Aufruf ist wie folgt:

python nlp_bertopic.py --input c:/PythonProjekt/meldungen_test.json --out_txt c:/PythonProjekt/meldungen.txt --out_csv c:/PythonProjekt/themen_uebersicht.csv --out_png c:/PythonProjekt/themen_haeufigkeit.png --out_top c:/PythonProjekt/top_themen.txt

python nlp_tfidf.py --input c:/PythonProjekt/meldungen_test.json --out_txt c:/PythonProjekt/meldungen.txt --out_csv c:/PythonProjekt/themen_uebersicht.csv --out_png c:/PythonProjekt/themen_haeufigkeit.png --out_top c:/PythonProjekt/top_themen.txt

 optionale Parameter 
  --topics 8                                        
  --label_words 4                                      
  --top_words 12                                       
  --spellcheck                                             

Durch --spellcheck wird die Rechtschreibung aktiviert.
Die genutzten Bibliotheken und deren Version sind in der Requirements.txt enthalten.
