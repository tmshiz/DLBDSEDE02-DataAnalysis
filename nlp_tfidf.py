# DataAnalysis
# Tim Schmitz
# 26.08.2025

import json
import re
import argparse
import time
from pathlib import Path
import nltk
import spacy
from nltk.corpus import stopwords
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Standard Speicherort definieren
desktop = Path.home() / "Desktop"

# Timer starten 
start_time = time.time()

# Parameter
parser = argparse.ArgumentParser(description="Themenmodellierung mit TF-IDF + LDA (DE)")
parser.add_argument("--input",      type=str, required=True,  help="Pfad zur Eingabedatei (JSON)")
parser.add_argument("--out_txt",    type=str, default=str(desktop / "Vorverarbeitung.txt"),  help="Speicherort für Übersicht der Vorverarbeitung")
parser.add_argument("--out_png",    type=str, default=str(desktop / "Themenübersicht.png"),  help="Speicherort für Plot Ausgabe")
parser.add_argument("--out_top",    type=str, default=str(desktop / "Themenübersicht.txt"),  help="Speicherort für Übersicht der Themenübersicht")
parser.add_argument("--spellcheck", action="store_true",      help="Aktiviere Rechtschreibkorrektur (pyspellchecker)")
parser.add_argument("--topics",     type=int, default=10,     help="Anzahl LDA-Themen (Default: 10)")
parser.add_argument("--label_words", type=int, default=3,   help="Anzahl Wörter pro Thema im Label (Default: 3)")
parser.add_argument("--top_words",  type=int, default=10,   help="Top-Wörter pro Thema (Default: 10)")
args = parser.parse_args()

# Pfade prüfen
for path in [args.out_txt, args.out_png, args.out_top]:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

# Stopwörter konfigurieren
nltk.download('stopwords', quiet=True)
STOP_DE = set(stopwords.words('german'))
nlp = spacy.load("de_core_news_sm")

# Rechtschreibung optional
spell = None
if args.spellcheck:
    from spellchecker import SpellChecker
    spell = SpellChecker(language="de")

def korrigiere_wort(w: str) -> str:
    if not spell or not w or not w.isalpha():
        return w
    if len(w) < 2 or len(w) > 30:
        return w
    if spell.known([w]):
        return w
    try:
        kandidaten = spell.candidates(w) or set()
    except Exception:
        kandidaten = set()
    if len(kandidaten) == 1:
        return next(iter(kandidaten))
    try:
        corr = spell.correction(w)
        if corr and isinstance(corr, str) and corr.isalpha():
            return corr
    except Exception:
        pass
    return w

# Funktion Vorverarbeitung
def bereinige_text(text: str) -> str:
    text = re.sub(r'[^a-zA-ZäöüÄÖÜß\s]', ' ', text)
    doc = nlp(text)
    toks = []
    for t in doc:
        if t.is_alpha:
            lemma = t.lemma_.lower()
            if lemma and lemma not in STOP_DE:
                toks.append(korrigiere_wort(lemma))
    return " ".join(toks)

# JSON einlesen
with open(args.input, "r", encoding="utf-8") as f:
    daten = json.load(f)

# Vorverarbeitung
bereinigte_texte = []
for eintrag in daten.get("index", []):
    for feld in ("betreff", "sachverhalt"):
        inhalt = eintrag.get(feld, "")
        if isinstance(inhalt, str) and inhalt.strip():
            bereinigte_texte.append(bereinige_text(inhalt))

if not bereinigte_texte:
    raise ValueError("Keine gültigen Texte nach Vorverarbeitung gefunden.")

# bereinigte Texte speichern
with open(args.out_txt, "w", encoding="utf-8") as f_out:
    for zeile in bereinigte_texte:
        f_out.write(zeile + '\n')
print(f"[INFO] Bereinigte Texte gespeichert: {args.out_txt}")

# Vektorisierung
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2), 
    min_df=2,
    max_df=0.95,
    lowercase=False
)
X = vectorizer.fit_transform(bereinigte_texte)
vocab = np.array(vectorizer.get_feature_names_out())

lda = LDA(
    n_components=args.topics,
    learning_method="batch",
    random_state=42
)
doc_topic = lda.fit_transform(X)
topic_word = lda.components_

# Top-Wörter pro Thema
def top_words_for_topic(t_idx: int, k: int):
    scores = topic_word[t_idx]
    top_idx = np.argsort(scores)[::-1][:k]
    return vocab[top_idx].tolist()

# Harte Dokument-Zuordnung
hard_topics = doc_topic.argmax(axis=1)

# topic_info DataFrame wie zuvor
rows = []
for t in range(args.topics):
    count = int((hard_topics == t).sum())
    label = ", ".join(top_words_for_topic(t, max(1, args.label_words)))
    rows.append({"Topic": t, "Count": count, "Label": label})

topic_info = pd.DataFrame(rows).sort_values("Count", ascending=False).reset_index(drop=True)

# Top-Wörtern
TWK = max(1, args.top_words)
topic_info[f"TopWords{TWK}"] = topic_info["Topic"].apply(
    lambda t: ", ".join(top_words_for_topic(int(t), TWK))
)

# Plot speichern
plt.figure(figsize=(12, 6))
plt.bar(topic_info['Label'], topic_info['Count'], color='steelblue')
plt.title("Häufigkeit der Themen (TF-IDF + LDA)")
plt.xlabel("Thema (Label)")
plt.ylabel("Anzahl der Dokumente")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(args.out_png, dpi=150)
plt.close()
print(f"[INFO] Plot gespeichert: {args.out_png}")

# Top-Themen TXT speichern
N = min(10, len(topic_info))
lines = [f"Top {N} Themen (TF-IDF + LDA)\n", "=" * 40 + "\n"]
for _, row in topic_info.head(N).iterrows():
    tid = int(row["Topic"])
    label = row["Label"]
    count = int(row["Count"])
    words = ", ".join(top_words_for_topic(tid, TWK))
    lines += [
        f"Topic {tid} | {label}\n",
        f"Count: {count}\n",
        f"Top-Wörter ({TWK}): {words}\n",
        "-" * 40 + "\n"
    ]
with open(args.out_top, "w", encoding="utf-8") as f_top:
    f_top.writelines([l if l.endswith("\n") else l + "\n" for l in lines])
print(f"[INFO] Top-Themen gespeichert: {args.out_top}")

# Timer stoppen
laufzeit = time.time() - start_time
print(f"[INFO] Scriptlaufzeit: {laufzeit:.2f} Sekunden")