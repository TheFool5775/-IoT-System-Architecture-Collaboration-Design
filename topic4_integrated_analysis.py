"""
Topic 4: Integrated syntactic + semantic analysis of social media posts
=======================================================================
Pipeline (Figure 1 in the report):
    post -> spaCy dependency parse -> phrase extraction -> TextBlob polarity
         -> negation check -> sentiment per target keyword

Features extracted with dependency relations:
    1. Noun phrases                      (doc.noun_chunks)
    2. Adjective-noun pairs              (dep = amod)
    3. Subject-complement pairs          (nsubj + acomp, e.g. "movie was bad")
    4. Verb-object pairs                 (dobj, e.g. "lost luggage")

Setup (once):
    pip install spacy textblob pandas matplotlib
    python -m spacy download en_core_web_sm
    python -m textblob.download_corpora

Run:
    python topic4_integrated_analysis.py                # built-in 30 posts
    python topic4_integrated_analysis.py posts.csv      # CSV with a "text" column
"""

import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from textblob import TextBlob

POS_THRESHOLD = 0.1    # post polarity above this  -> positive
NEG_THRESHOLD = -0.1   # post polarity below this  -> negative

# ----------------------------------------------------------------------------
# Dataset: 30 social media posts (replace with your own if you have them)
# ----------------------------------------------------------------------------
POSTS = [
    "The camera is amazing and the screen is beautiful!",
    "My phone battery is terrible, it dies by noon.",
    "Best coffee in town, I love this place!",
    "Stuck in terrible traffic again, I hate this commute.",
    "Great teamwork today, proud of the whole crew.",
    "The airline lost my luggage and nobody cares.",
    "This new update ruined the app.",
    "Amazing weather this weekend, perfect for the beach.",
    "The internet is slow and unreliable today.",
    "What a wonderful job the support team did!",
    "The meeting is scheduled for 3pm tomorrow.",
    "Heading to the market to buy vegetables.",
    "Delicious dinner with great friends tonight.",
    "Can't stand the new layout.",
    "Our flight was smooth and the crew was friendly.",
    "Hiyo movie ilikuwa poa sana, we should watch it again.",
    "Long queue at the clinic, a complete waste of time.",
    "The service wasn't good at all.",
    "I love my new job, the people are so kind.",
    "Ugh, the queue at the bank is endless.",
    "Just finished a long day, going to bed.",
    "lol that movie was so bad it was actually funny.",
    "Not happy with the delivery, the box was damaged.",
    "The coffee was cold and the waiter was rude.",
    "Beautiful sunset tonight, feeling grateful.",
    "The new phone is expensive but the battery is excellent.",
    "Terrible customer service, I will never shop here again.",
    "The concert starts at 8pm at the stadium.",
    "What a boring and slow day at work.",
    "Wonderful teamwork from everyone on the project!",
]

nlp = spacy.load("en_core_web_sm")


# ----------------------------------------------------------------------------
# Semantic helpers
# ----------------------------------------------------------------------------
def word_polarity(word: str) -> float:
    """TextBlob polarity (-1..+1) of a single word."""
    return TextBlob(word).sentiment.polarity


def is_negated(token) -> bool:
    """True if a negation (dep = neg) is attached to the token, or to its head verb
    (e.g. 'wasn't good': 'n't' is attached to 'was', the head of 'good')."""
    if any(child.dep_ == "neg" for child in token.children):
        return True
    return token.dep_ == "acomp" and any(c.dep_ == "neg" for c in token.head.children)


def scored(token) -> float:
    """Polarity of an opinion word, flipped when the parse shows a negation."""
    score = word_polarity(token.lemma_.lower())
    if score == 0:                       # try the surface form if the lemma is unknown
        score = word_polarity(token.text.lower())
    return -score if (score != 0 and is_negated(token)) else score


# ----------------------------------------------------------------------------
# Syntactic feature extraction (one post at a time)
# ----------------------------------------------------------------------------
def extract_features(doc):
    noun_phrases, amod, subj_comp, verb_obj = [], [], [], []

    for chunk in doc.noun_chunks:
        if chunk.root.pos_ != "PRON":                       # skip I, it, we ...
            noun_phrases.append(chunk.text.lower())

    for tok in doc:
        # adjective + noun: "terrible traffic"
        if tok.dep_ == "amod" and tok.head.pos_ in ("NOUN", "PROPN"):
            amod.append((tok.head.lemma_.lower(), tok))

        # subject + complement: "movie was bad"
        elif tok.dep_ == "acomp":
            for sub in (c for c in tok.head.children if c.dep_ == "nsubj"):
                if sub.pos_ in ("NOUN", "PROPN"):
                    subj_comp.append((sub.lemma_.lower(), tok))

        # verb + direct object: "lost luggage"
        elif tok.dep_ == "dobj" and tok.pos_ in ("NOUN", "PROPN"):
            verb_obj.append((tok.lemma_.lower(), tok.head))

    return noun_phrases, amod, subj_comp, verb_obj


def classify(polarity: float) -> str:
    if polarity > POS_THRESHOLD:
        return "Positive"
    if polarity < NEG_THRESHOLD:
        return "Negative"
    return "Neutral"


# ----------------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------------
def analyse(posts):
    post_rows, pair_rows = [], []
    counts = {"noun_phrases": 0, "adj_noun": 0, "subj_comp": 0, "verb_obj": 0}

    for i, text in enumerate(posts, start=1):
        doc = nlp(text)
        nps, amod, sc, vo = extract_features(doc)
        counts["noun_phrases"] += len(nps)
        counts["adj_noun"] += len(amod)
        counts["subj_comp"] += len(sc)
        counts["verb_obj"] += len(vo)

        polarity = TextBlob(text).sentiment.polarity          # post-level score
        post_rows.append({"post": i, "text": text, "polarity": round(polarity, 3),
                          "sentiment": classify(polarity), "noun_phrases": "; ".join(nps)})

        for kind, pairs in (("adj-noun", amod), ("subj-comp", sc), ("verb-obj", vo)):
            for keyword, opinion in pairs:
                pair_rows.append({"post": i, "type": kind, "keyword": keyword,
                                  "opinion_word": opinion.text.lower(),
                                  "negated": is_negated(opinion),
                                  "score": round(scored(opinion), 3)})

    return pd.DataFrame(post_rows), pd.DataFrame(pair_rows), counts


def plot_keywords(pairs: pd.DataFrame, path="keyword_sentiment.png"):
    """Mean polarity of opinion words per keyword (keywords with a non-zero score)."""
    means = pairs.groupby("keyword")["score"].mean()
    means = means[means != 0].sort_values()
    if means.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(means) + 1.5))
    ax.barh(means.index, means.values,
            color=["#c0504d" if v < 0 else "#3578b8" for v in means.values])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Mean TextBlob polarity of opinion words attached to the keyword")
    ax.set_title("Sentiment by target keyword (from dependency relations)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    posts = POSTS
    if len(sys.argv) > 1:
        posts = pd.read_csv(sys.argv[1])["text"].dropna().astype(str).tolist()

    post_df, pair_df, counts = analyse(posts)

    print("\n=== Post-level sentiment ===")
    for _, r in post_df.iterrows():
        print(f"{r.post:>2}. {r.sentiment:<8} {r.polarity:+.2f}  {r.text}")

    vc = post_df["sentiment"].value_counts()
    print(f"\nPositive: {vc.get('Positive', 0)} | Neutral: {vc.get('Neutral', 0)} | "
          f"Negative: {vc.get('Negative', 0)} | Mean polarity: {post_df['polarity'].mean():+.2f}")

    print("\n=== Extracted features ===")
    print(f"Noun phrases: {counts['noun_phrases']} | Adjective-noun pairs: {counts['adj_noun']} | "
          f"Subject-complement pairs: {counts['subj_comp']} | Verb-object pairs: {counts['verb_obj']}")

    if not pair_df.empty:
        print("\n=== Opinion pairs ===")
        print(pair_df.to_string(index=False))
        print("\n=== Mean sentiment per keyword ===")
        print(pair_df.groupby("keyword")["score"].agg(["mean", "count"]).round(2)
              .sort_values("mean").to_string())
        plot_keywords(pair_df)
        pair_df.to_csv("opinion_pairs.csv", index=False)

    post_df.to_csv("post_sentiment.csv", index=False)
    print("\nSaved: post_sentiment.csv, opinion_pairs.csv, keyword_sentiment.png")


if __name__ == "__main__":
    main()
