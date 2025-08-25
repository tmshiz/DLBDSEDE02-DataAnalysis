# DataAnalysis #24.08.2025 21:00 Version 2.3 
import json
import re
import argparse
import nltk
import spacy
import time
from nltk.corpus import stopwords
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ---------------- Timer Start ----------------
start_time = time.time()

# ---------------- Argumente ----------------
parser = argparse.ArgumentParser(description="Themenmodellierung mit BERTopic (DE)")
parser.add_argument("--input",    type=str, required=True,  help="Pfad zur Eingabedatei (JSON)")
parser.add_argument("--out_txt",  type=str, required=True,  help="Pfad zur Ausgabedatei (bereinigte Texte, TXT)")
parser.add_argument("--out_csv",  type=str, required=True,  help="Pfad zur Ausgabedatei (Themenübersicht, CSV)")
parser.add_argument("--out_png",  type=str, required=True,  help="Pfad zur Ausgabedatei (Plot, PNG)")
parser.add_argument("--out_top",  type=str, required=True,  help="Pfad zur Ausgabedatei (Top-Themen, TXT)")
parser.add_argument("--spellcheck", action="store_true",    help="Aktiviere Rechtschreibkorrektur (pyspellchecker)")
parser.add_argument("--topics", type=int, default=10,     help="Zielanzahl der Themen (BERTopic nr_topics). Beispiel: --topics 5")
parser.add_argument("--label_words", type=int, default=3,   help="Anzahl Wörter pro Thema im Label (Default: 3)")
parser.add_argument("--top_words",  type=int, default=10,   help="Top-Wörter pro Thema für CSV/TXT (Default: 10)")

args = parser.parse_args()

# ---------------- NLP Config ----------------
nltk.download('stopwords', quiet=True)
stopwoerter = set(stopwords.words('german'))
nlp = spacy.load("de_core_news_sm")

# Rechtschreibung optional
spell = None
if args.spellcheck:
    from spellchecker import SpellChecker
    spell = SpellChecker(language="de")

# ---------------- Pfade prüfen ----------------
for path in [args.out_txt, args.out_csv, args.out_png, args.out_top]:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

# ---------------- Hilfsfunktionen ----------------
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
        if corr and isinstance(corr, str) and corr.isalpha() and corr != w:
            return corr
    except Exception:
        pass
    return w

def bereinige_text(text: str) -> str:
    text = re.sub(r'[^a-zA-ZäöüÄÖÜß\s]', ' ', text)
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_alpha:
            lemma = token.lemma_.lower()
            if lemma and lemma not in stopwoerter:
                tokens.append(korrigiere_wort(lemma))
    return ' '.join(tokens)

# ---------------- Daten einlesen ----------------
with open(args.input, 'r', encoding='utf-8') as f:
    daten = json.load(f)

bereinigte_texte = []
for eintrag in daten.get("index", []):
    for feld in ("betreff", "sachverhalt"):
        inhalt = eintrag.get(feld, "")
        if isinstance(inhalt, str) and inhalt.strip():
            bereinigte_texte.append(bereinige_text(inhalt))

# ---------------- Speichern: TXT (bereinigte Texte) ----------------
with open(args.out_txt, 'w', encoding='utf-8') as f_out:
    for zeile in bereinigte_texte:
        f_out.write(zeile + '\n')
print(f"[INFO] Bereinigte Texte gespeichert: {args.out_txt}")

# ---------------- BERTopic ----------------
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

topic_model = BERTopic(
    embedding_model=embedding_model,
    language="german",
    nr_topics=args.topics
)

topics, probs = topic_model.fit_transform(bereinigte_texte)

# Themenübersicht aufbereiten
topic_info = topic_model.get_topic_info()
topic_info = topic_info[topic_info['Topic'] != -1].sort_values(by="Count", ascending=False).copy()

# Labels
labels = topic_model.generate_topic_labels(nr_words=args.label_words)
def safe_label(x):
    try:
        return labels[x] if isinstance(labels, dict) else labels[x]
    except Exception:
        return f"Topic {x}"
topic_info["Label"] = topic_info["Topic"].apply(safe_label)

# Top-Wörter in CSV aufnehmen
topics_dict = topic_model.get_topics()
def top_words_str(tid, k=args.top_words):
    pairs = topics_dict.get(tid, [])[:k]
    return ", ".join([w for w, _ in pairs]) if pairs else ""

topic_info[f"TopWords{args.top_words}"] = topic_info["Topic"].apply(lambda t: top_words_str(t, args.top_words))

# ---------------- Plot speichern ----------------
plt.figure(figsize=(12, 6))
plt.bar(topic_info['Label'], topic_info['Count'], color='steelblue')
plt.title("Häufigkeit der Themen")
plt.xlabel("Thema (automatisch generiert)")
plt.ylabel("Anzahl der Dokumente")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(args.out_png, dpi=150)
plt.close()
print(f"[INFO] Plot gespeichert: {args.out_png}")

# ---------------- CSV speichern ----------------
topic_info.to_csv(args.out_csv, index=False, encoding="utf-8")
print(f"[INFO] Themenübersicht gespeichert: {args.out_csv}")

# ---------------- Top-Themen TXT speichern ----------------
N = min(10, len(topic_info))
lines = [f"Top {N} Themen\n", "=" * 40 + "\n"]
for _, row in topic_info.head(N).iterrows():
    tid = row["Topic"]
    label = row["Label"]
    count = int(row["Count"])
    words = top_words_str(tid, args.top_words)
    lines += [
        f"Topic {tid} | {label}\n",
        f"Count: {count}\n",
        f"Top-Wörter ({args.top_words}): {words}\n",
        "-" * 40 + "\n"
    ]
with open(args.out_top, "w", encoding="utf-8") as f_top:
    f_top.writelines([l if l.endswith("\n") else l + "\n" for l in lines])
print(f"[INFO] Top-Themen gespeichert: {args.out_top}")

# ---------------- Timer Ende ----------------
laufzeit = time.time() - start_time
print(f"[INFO] Scriptlaufzeit: {laufzeit:.2f} Sekunden")
