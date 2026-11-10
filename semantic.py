#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install textblob pandas matplotlib')


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob


# In[6]:


# Sample social media tweets

tweets = [
    "I love this product! It is amazing.",
    "The service was excellent and I am very happy.",
    "This is the best experience ever!",
    "I really enjoy using this application.",
    "The food was delicious and wonderful.",

    "I hate this product. It is terrible.",
    "The service was very poor and disappointing.",
    "I am unhappy with the quality.",
    "This application is frustrating and slow.",
    "The experience was awful.",

    "The product arrived yesterday.",
    "The company released a new product.",
    "I bought a phone this morning.",
    "The meeting starts at 10 AM.",
    "The application was updated today."
]

# Create a DataFrame
df = pd.DataFrame(tweets, columns=["Tweet"])

# Display the dataset
df.head(15)


# In[7]:


# Calculate polarity for each tweet

df["Polarity"] = df["Tweet"].apply(
    lambda tweet: TextBlob(tweet).sentiment.polarity
)

# Display polarity scores
df[["Tweet", "Polarity"]]


# In[8]:


# Function to classify sentiment

def classify_sentiment(polarity):
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply classification
df["Sentiment"] = df["Polarity"].apply(classify_sentiment)

# Display results
print("SENTIMENT ANALYSIS RESULTS")
display(df)


# In[9]:


# Calculate overall sentiment score

overall_score = df["Polarity"].mean()

print("Overall Sentiment Score:", round(overall_score, 3))

# Interpret overall sentiment

if overall_score > 0:
    print("Overall Sentiment: Positive")
elif overall_score < 0:
    print("Overall Sentiment: Negative")
else:
    print("Overall Sentiment: Neutral")


# In[10]:


# Count tweets in each sentiment category

sentiment_counts = df["Sentiment"].value_counts().reindex(
    ["Positive", "Negative", "Neutral"],
    fill_value=0
)

# Calculate percentages

sentiment_percentages = (
    sentiment_counts / len(df) * 100
)

print("SENTIMENT COUNTS")
print(sentiment_counts)

print("\nSENTIMENT PERCENTAGES")
print(sentiment_percentages.round(2))


# In[11]:


# Create a bar chart

plt.figure(figsize=(8, 5))

sentiment_counts.plot(kind="bar")

plt.title("Social Media Sentiment Analysis")
plt.xlabel("Sentiment Category")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=0)
plt.tight_layout()

plt.show()


# In[12]:


# Display final analysis summary

print("========== SENTIMENT ANALYSIS SUMMARY ==========")

print("Total Tweets:", len(df))

print("\nSentiment Distribution:")
for sentiment, count in sentiment_counts.items():
    percentage = sentiment_percentages[sentiment]
    print(f"{sentiment}: {count} tweets ({percentage:.2f}%)")

print("\nOverall Sentiment Score:", round(overall_score, 3))

print("================================================")


# In[ ]:




