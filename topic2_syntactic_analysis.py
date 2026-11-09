"""
Topic 2: Syntactic Analysis of Social Media Text
Tasks: (1) Part-of-Speech (POS) tagging  (2) Dependency parsing

Setup:
    pip install spacy pandas matplotlib
    python -m spacy download en_core_web_sm
Run:
    python topic2_syntactic_analysis.py
"""

import re
from collections import Counter

import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

# ---------------------------------------------------------------
# 1. Sample dataset: social media posts (mix of clean and noisy)
# ---------------------------------------------------------------
posts = [
    "The new phone camera is absolutely amazing!",
    "Just landed in Nairobi and the weather is lovely.",
    "I can't believe the airline lost my luggage again.",
    "Customer service solved my problem quickly. Great job!",
    "This update ruined the app and nobody fixed it.",
    "omg the concert last night was sooo good lol",
    "Traffic on Mombasa Road is terrible today #nightmare",
    "@SafaricomPLC why is my internet so slow???",
    "Loving the new battery life, best upgrade ever :)",
    "The delivery arrived late and the box was damaged.",
]


# ---------------------------------------------------------------
# 2. Light cleaning: keep hashtags/mentions as normal words so the
#    parser sees them; remove URLs only.
# ---------------------------------------------------------------
def clean(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    return text.strip()


docs = [nlp(clean(p)) for p in posts]

# ---------------------------------------------------------------
# 3. POS tagging
#    token.pos_ = coarse Universal POS tag, token.tag_ = fine-grained
#    Penn Treebank tag
# ---------------------------------------------------------------
print("=" * 70)
print("PART 1: POS TAGGING")
print("=" * 70)

pos_rows = []
for i, doc in enumerate(docs, 1):
    print(f"\nPost {i}: {doc.text}")
    print(f"{'TOKEN':<14}{'UPOS':<8}{'PENN TAG':<10}{'EXPLANATION'}")
    for tok in doc:
        print(f"{tok.text:<14}{tok.pos_:<8}{tok.tag_:<10}{spacy.explain(tok.tag_)}")
        pos_rows.append(
            {"post": i, "token": tok.text, "upos": tok.pos_, "tag": tok.tag_}
        )

pos_df = pd.DataFrame(pos_rows)
pos_df.to_csv("topic2_pos_tags.csv", index=False)

print("\nPOS distribution across all posts:")
print(pos_df["upos"].value_counts().to_string())

# ---------------------------------------------------------------
# 4. Dependency parsing
#    token.dep_ = relation to head, token.head = governing word
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("PART 2: DEPENDENCY PARSING")
print("=" * 70)

dep_rows = []
for i, doc in enumerate(docs, 1):
    print(f"\nPost {i}: {doc.text}")
    print(f"{'TOKEN':<14}{'DEP':<10}{'HEAD':<14}{'CHILDREN'}")
    for tok in doc:
        children = [c.text for c in tok.children]
        print(f"{tok.text:<14}{tok.dep_:<10}{tok.head.text:<14}{children}")
        dep_rows.append(
            {
                "post": i,
                "token": tok.text,
                "dep": tok.dep_,
                "head": tok.head.text,
            }
        )

dep_df = pd.DataFrame(dep_rows)
dep_df.to_csv("topic2_dependencies.csv", index=False)

print("\nMost common dependency relations:")
print(dep_df["dep"].value_counts().head(10).to_string())

# ---------------------------------------------------------------
# 5. Useful syntactic extraction: subject - verb - object triples
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("SUBJECT-VERB-OBJECT TRIPLES")
print("=" * 70)
for i, doc in enumerate(docs, 1):
    for tok in doc:
        if tok.pos_ in ("VERB", "AUX") and tok.dep_ in ("ROOT", "conj", "ccomp"):
            subj = [c.text for c in tok.children if c.dep_ in ("nsubj", "nsubjpass")]
            obj = [c.text for c in tok.children if c.dep_ in ("dobj", "attr", "acomp")]
            if subj or obj:
                print(f"Post {i}: {subj} -> {tok.lemma_} -> {obj}")

# ---------------------------------------------------------------
# 6. Dependency diagrams (SVG files for the report)
# ---------------------------------------------------------------
from spacy import displacy
from pathlib import Path

for idx in (0, 2, 5):  # a clean, a contraction, and a noisy post
    svg = displacy.render(docs[idx], style="dep", jupyter=False,
                          options={"compact": False, "distance": 110})
    Path(f"topic2_parse_post{idx + 1}.svg").write_text(svg, encoding="utf-8")

print("\nSaved: topic2_pos_tags.csv, topic2_dependencies.csv, parse SVGs")

# ---------------------------------------------------------------
# 7. Noise analysis: how often does the tagger label chat-speak oddly?
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("NOISY TEXT CHECK (Post 6, 7, 8)")
print("=" * 70)
for idx in (5, 6, 7):
    print([(t.text, t.pos_) for t in docs[idx]])
