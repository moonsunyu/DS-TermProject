import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv("data/movies_original_dataset.csv")

# 백업
df_original = df.copy()
df_original.to_csv("data/movies_backup.csv", index=False)

# 결측치가 존재하는 컬럼
missing_columns = df_original.columns[df.isnull().any()]
print("Columns with missing values:")
for col in missing_columns:
    print(f"- {col}: {df[col].isnull().sum()} entries missing")

print("\nDataset size:", df.shape)

missing_both_count = df[df['budget'].isnull() & df['gross'].isnull()].shape[0]

# 예산과 수익이 모두 없는 영화 개수
print("Number of movies missing both budget and gross:", missing_both_count) # 128
# 모두 제거
df = df.dropna(subset=['budget', 'gross'], how='all')
print("Remaining number of movies:", len(df))
print("Remaining missing values in 'budget':", df['budget'].isnull().sum())
print("Remaining missing values in 'gross':", df['gross'].isnull().sum())


# 여전히 결측치가 존재하는 컬럼
missing_columns = df.columns[df.isnull().any()]
print("\nColumns still containing missing values:")
for col in missing_columns:
    print(f"- {col}: {df[col].isnull().sum()} entries missing")

# - rating: 62 entries missing
# - score: 3 entries missing
# - votes: 3 entries missing
# - writer: 3 entries missing
# - star: 1 entries missing
# - country: 1 entries missing
# - budget: 2043 entries missing
# - gross: 61 entries missing
# - company: 14 entries missing
# - runtime: 2 entries missing

# 평점과 평가수가 모두 없는 영화 개수
missing_both_count2 = df[df['score'].isnull() & df['votes'].isnull()].shape[0]
print("Number of movies missing both score and votes: ",missing_both_count2)
df = df.dropna(subset=['score', 'votes'], how='all')

# 나머지 결측치 처리

# rating: replace missing values with 'Not Rated'
df['rating'].fillna('Not Rated', inplace=True)

# writer: replace missing values with 'Unknown'
df['writer'].fillna('Unknown', inplace=True)

# star: replace missing values with 'Unknown'
df['star'].fillna('Unknown', inplace=True)

# country: replace missing values with the most frequent value
df['country'].fillna(df['country'].mode()[0], inplace=True)

# company: replace missing values with 'Unknown'
df['company'].fillna('Unknown', inplace=True)

# runtime: replace missing values with the column median
df['runtime'].fillna(df['runtime'].median(), inplace=True)

# gross: drop rows with missing gross value 
df = df.dropna(subset=['gross'])

# budget: drop rows with missing budget value 
df = df.dropna(subset=['budget'])

print("\nAfter Replacing NaN:")
for col in missing_columns:
    print(f"- {col}: {df[col].isnull().sum()} entries missing")

df.to_csv("data/movies_cleaned.csv", index=False)
print("Saved cleaned file to movies_cleaned.csv file")

# 점검용 코드
df2 = pd.read_csv("movies_cleaned.csv")
print(df2.shape)
print(df2.columns)
print(df2.dtypes)
print("\nMissing values")
missing_info = df2.isnull().sum().sort_values(ascending=False)
missing_ratio = df2.isnull().mean().sort_values(ascending=False)
print(pd.concat([missing_info, missing_ratio], axis=1, keys=["Missing Count", "Missing Ratio"]))
