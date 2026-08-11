import pandas as pd
df = pd.read_csv("data/sentiment_dataset.csv")
print("Dataset shape:", df.shape)

# Summary statistics for numeric columns
print("\nSummary statistics (Likes and Retweets):")
print(df[["Likes", "Retweets"]].describe())
import matplotlib.pyplot as plt

# Histogram for Likes
plt.figure()
df["Likes"].hist(bins=10, color='darkgreen')
plt.title("Distribution of Likes", fontweight="bold")
plt.xlabel("Likes")
plt.ylabel("Number of Posts")
plt.savefig("likes_histogram.png")
plt.grid(False)
plt.show()

# Boxplot for Likes 
plt.figure()
df["Likes"].plot(kind="box")
plt.title("Boxplot of Likes", fontweight="bold")
plt.ylabel("Likes")
plt.savefig("likes_boxplot.png")
plt.grid(False)
plt.show()

# Histogram for Retweets
plt.figure()
df["Retweets"].hist(bins=10, color="lightgreen")
plt.title("Distribution of Retweets", fontweight="bold")
plt.xlabel("Retweets")
plt.ylabel("Number of Posts")
plt.savefig("retweets_histogram.png")
plt.grid(False)
plt.show()

# Boxplot for Retweets
plt.figure()
df["Retweets"].plot(kind="box")
plt.title("Boxplot of Retweets", fontweight="bold")
plt.ylabel("Retweets")
plt.savefig("retweets_boxplot.png")
plt.grid(False)
plt.show()

# Scatter plot for Likes vs Retweets
plt.figure()
plt.scatter(df["Likes"], df["Retweets"], color="darkgreen")
plt.title("Likes vs Retweets", fontweight="bold")
plt.xlabel("Likes")
plt.ylabel("Retweets")
plt.savefig("likes_vs_retweets_scatter.png")
plt.grid(False)
plt.show()

# Calculate correlation between Likes and Retweets
correlation = df["Likes"].corr(df["Retweets"])
print("\nCorrelation between Likes and Retweets:", correlation)