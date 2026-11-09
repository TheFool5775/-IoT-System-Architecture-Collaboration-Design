import nltk
import pandas as pd

from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet
from nltk import pos_tag

# 1. Download required NLP resources
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')

# 2. Create sample social media dataset
data = {
    'Post': [
        "I am loving the new technology trends!",
        "Students are studying hard for their exams.",
        "The players were running quickly yesterday.",
        "Social media users share amazing pictures.",
        "Businesses are improving their marketing strategies."
    ]
}

df = pd.DataFrame(data)

print("\nORIGINAL SOCIAL MEDIA DATASET")
print(df.to_string(index=False))

# 3. Initialize NLP tools
tokenizer = TweetTokenizer()
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# 4. TOKENIZATION
df['Tokenized'] = df['Post'].apply(tokenizer.tokenize)

print("\nTOKENIZATION RESULTS")
for post in df['Tokenized']:
    print(post)

# 5. POS mapping for lemmatization
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# 6. LEMMATIZATION
def lemmatize_text(text):
    tokens = tokenizer.tokenize(text)
    tagged_words = pos_tag(tokens)

    result = []

    for word, tag in tagged_words:
        if word.isalpha():
            result.append(
                lemmatizer.lemmatize(
                    word.lower(),
                    get_wordnet_pos(tag)
                )
            )
        else:
            result.append(word)

    return result

df['Lemmatized'] = df['Post'].apply(lemmatize_text)

print("\nLEMMATIZATION RESULTS")
for post in df['Lemmatized']:
    print(post)

# 7. STEMMING
def stem_text(text):
    tokens = tokenizer.tokenize(text)

    return [
        stemmer.stem(word.lower())
        if word.isalpha() else word
        for word in tokens
    ]

df['Stemmed'] = df['Post'].apply(stem_text)

print("\nSTEMMING RESULTS")
for post in df['Stemmed']:
    print(post)

# 8. DISPLAY COMPLETE PROCESSED DATASET
print("\nCOMPLETE PROCESSED DATASET")
print(df.to_string(index=False))
