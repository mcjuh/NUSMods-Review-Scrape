import pandas as pd

data = pd.read_csv('nus_module_sentiments - Copy.csv')
data['Total Reviews'] = data['Positive Comments'] + data['Negative Comments']
reviewed_modules = data[data['Total Reviews'] >= 5]
reviewed_modules['Module Prefix'] = reviewed_modules['Module Code'].str.extract(r'^([A-Za-z]+)')

prefix_reviews = reviewed_modules.groupby('Module Prefix')['Total Reviews'].sum().sort_values(ascending=False)
top_reviewed = reviewed_modules.sort_values('Total Reviews', ascending=False).head(10)

mean_sentiment = data['Aggregated Sentiment Score'].mean()
prefix_sentiment = reviewed_modules.groupby('Module Prefix')['Aggregated Sentiment Score'].mean().sort_values()

print("Top reviewed module prefixes:\n", prefix_reviews)
print("\nTop 10 reviewed modules:\n", top_reviewed)

print(f"Average Sentiment Score: {mean_sentiment}")
print("\nAverage Sentiment Score by Prefix:\n", prefix_sentiment)