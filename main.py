import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from wordcloud import WordCloud

# Reading in the data
data = pd.read_csv('AmazonReview.csv')

# Dropping na's out of data
data.dropna(inplace=True)

# Renaming columns and setting sentiment
data.rename(columns={'Review': 'Review', 'Sentiment': 'Rating'}, inplace=True)
data.loc[data['Rating'] <= 3, 'Sentiment'] = 0
data.loc[data['Rating'] > 3, 'Sentiment'] = 1

# Cleaning reviews
stp_words = stopwords.words('english')

def clean_review(review):
    clean_review = ' '.join(word for word in review.split() if word not in stp_words and word.lower() not in ['br'])
    return clean_review


data['Review'] = data['Review'].apply(clean_review)

# Visualizing sentiment distribution
sentiment_counts = data['Sentiment'].value_counts()
sentiment_counts.plot(kind='bar')
plt.title('Sentiment Distribution')
plt.xticks([0, 1], ['Negative', 'Positive'], rotation=0)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig('SentimentDistribution.png')

# Creating word clouds for positive and negative reviews
positive_reviews = ' '.join(word for word in data['Review'][data['Sentiment'] == 1].astype(str))
negative_reviews = ' '.join(word for word in data['Review'][data['Sentiment'] == 0].astype(str))

positive_wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(positive_reviews)
negative_wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(negative_reviews)

# Plotting word clouds
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
ax1.imshow(negative_wordcloud, interpolation='bilinear')
ax1.axis('off')
ax1.set_title('Word Cloud for Negative Reviews', fontsize=20)
ax2.imshow(positive_wordcloud, interpolation='bilinear')
ax2.axis('off')
ax2.set_title('Word Cloud for Positive Reviews', fontsize=20)
plt.savefig('WordClouds.png')

# Fitting logistic regression model
cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(data['Review']).toarray()
y = data['Sentiment']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)

# Evaluating model performance
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=[False, True])
cmd.plot()
plt.title('Confusion Matrix')
plt.savefig('ConfusionMatrix.png')

# Accuracy score of 81.6%