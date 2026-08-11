import pandas as pd

# Load the dataset
df = pd.read_csv("sentiment_dataset.csv")

# Display the first five rows
print(df.head())

# Display the column names
print(df.columns)

# Display info about the dataset
print(df.info())
print (df['Sentiment'].unique())
from textblob import TextBlob

# Use TextBlob to demonstrate sentiment scoring on the Hashtags text
def get_polarity(text):
    return TextBlob(str(text)).sentiment.polarity

df['Hashtag_Polarity'] = df['Hashtags'].apply(get_polarity)
print(df[['Hashtags', 'Hashtag_Polarity']].head())

# Show distribution of the existing Sentiment labels
print('\nSentiment value counts:')
print(df['Sentiment'].value_counts())
# Visualize sentiment distribution as a bar chart
import matplotlib.pyplot as plt

df['Sentiment'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Sentiment Categories', fontsize=16, fontweight='bold', color='blue')
plt.xlabel('Sentiment', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('sentiment_distribution.png')
plt.show()
# Create a word cloud from the hashtags
from wordcloud import WordCloud

all_hashtags = ' '.join(df['Hashtags'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_hashtags)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Hashtags', fontsize=16, fontweight='bold', color='blue')
plt.tight_layout()
plt.savefig('hashtag_wordcloud.png')
plt.show()