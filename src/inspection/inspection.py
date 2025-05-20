import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/movies_original_dataset.csv")

print(df.shape)
print(df.columns)
print(df.dtypes)

# Numerical / Categorical Columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

print("Numerical Columns:")
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(numeric_cols)

print("\nCategorical Columns:")
categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
print(categorical_cols)

print("\nMissing values")
missing_info = df.isnull().sum().sort_values(ascending=False)
missing_ratio = df.isnull().mean().sort_values(ascending=False)
print(pd.concat([missing_info, missing_ratio], axis=1, keys=["Missing Count", "Missing Ratio"]))


print("\nNumerical column basic statistics (describe):")
print(df[numeric_cols].describe().T)

print("\nTop 10 Categorical Column Value Distributions:")
for col in categorical_cols:
    print(f"\n{col} value_counts (Top 10):")
    print(df[col].value_counts(dropna=False).head(10))



# Data Columns Visualization

sns.set(style="whitegrid")

# 1. Rating distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='rating', order=df['rating'].value_counts().index)
plt.title("Number of Movies by Rating")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 2. Top 10 Genres
plt.figure(figsize=(8, 6))
sns.countplot(data=df, y='genre', order=df['genre'].value_counts().index[:10])
plt.title("Top 10 Genres")
plt.xlabel("Count")
plt.ylabel("Genre")
plt.tight_layout()
plt.show()

# 3. Movies Released per Year
plt.figure(figsize=(10, 6))
year_counts = df['year'].value_counts().sort_index()
sns.lineplot(x=year_counts.index, y=year_counts.values)
plt.title("Number of Movies Released per Year")
plt.xlabel("Year")
plt.ylabel("Movie Count")
plt.tight_layout()
plt.show()

# 4. Score Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['score'].dropna(), bins=20, kde=True)
plt.title("Distribution of Movie Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 5. Vote Distribution (log scale)
plt.figure(figsize=(8, 6))
sns.histplot(df['votes'].dropna(), bins=30, log_scale=True)
plt.title("Distribution of Vote Counts (Log Scale)")
plt.xlabel("Votes")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 6. Budget Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['budget'].dropna(), bins=30, kde=True)
plt.title("Distribution of Movie Budgets")
plt.xlabel("Budget (USD)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 7. Gross Revenue Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['gross'].dropna(), bins=30, kde=True)
plt.title("Distribution of Gross Revenue")
plt.xlabel("Gross Revenue (USD)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 8. Runtime Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['runtime'].dropna(), bins=20, kde=True)
plt.title("Distribution of Movie Runtime")
plt.xlabel("Runtime (minutes)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 9. Top 10 Movie Production Countries
plt.figure(figsize=(8, 6))
top_countries = df['country'].value_counts().head(10)
sns.barplot(x=top_countries.values, y=top_countries.index)
plt.title("Top 10 Movie-Producing Countries")
plt.xlabel("Movie Count")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

# 10. Top 10 Production Companies
plt.figure(figsize=(8, 6))
top_companies = df['company'].value_counts().head(10)
sns.barplot(x=top_companies.values, y=top_companies.index)
plt.title("Top 10 Production Companies")
plt.xlabel("Movie Count")
plt.ylabel("Company")
plt.tight_layout()
plt.show()

# 11. Top 10 Directors
plt.figure(figsize=(8, 6))
top_directors = df['director'].value_counts().head(10)
sns.barplot(x=top_directors.values, y=top_directors.index)
plt.title("Top 10 Directors")
plt.xlabel("Movie Count")
plt.ylabel("Director")
plt.tight_layout()
plt.show()

# 12. Top 10 Writers
plt.figure(figsize=(8, 6))
top_writers = df['writer'].value_counts().head(10)
sns.barplot(x=top_writers.values, y=top_writers.index)
plt.title("Top 10 Writers")
plt.xlabel("Movie Count")
plt.ylabel("Writer")
plt.tight_layout()
plt.show()

# 13. Top 10 Stars
plt.figure(figsize=(8, 6))
top_stars = df['star'].value_counts().head(10)
sns.barplot(x=top_stars.values, y=top_stars.index)
plt.title("Top 10 Lead Actors/Actresses")
plt.xlabel("Movie Count")
plt.ylabel("Star")
plt.tight_layout()
plt.show()
