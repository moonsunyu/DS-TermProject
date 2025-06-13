import pandas as pd
import numpy as np

# Dataset
df = pd.read_csv("data/cleaned/movies_cleaned.csv")

# Remove year, released
df.drop(columns=['year', 'released'], inplace=True, errors='ignore')

# New column 'weighted score': score * log_votes
df['votes'] = df['votes'].apply(lambda x: max(x, 0))  # 음수 방지
df['log_votes'] = np.log1p(df['votes'])
df['weighted_score'] = df['score'] * df['log_votes']

# Classify runtime into short / normal / long categories
def runtime_category(rt):
    if rt < 90:
        return 'short'
    elif rt < 120:
        return 'normal'
    else:
        return 'long'
df['runtime_category'] = df['runtime'].apply(runtime_category)

# Fill with 'Others', excluding the top n
def create_top_n_column(df, col_name, top_n, new_col_name):
    top_values = df[col_name].value_counts().head(top_n).index
    df[new_col_name] = df[col_name].apply(lambda x: x if x in top_values else 'Others')
    return df

df = create_top_n_column(df, 'director', 10, 'director_top10')      # 상위 10 direcotr
df = create_top_n_column(df, 'writer', 10, 'writer_top10')          # 상위 10 writer
df = create_top_n_column(df, 'star', 30, 'star_top30')              # 상위 30 star
df = create_top_n_column(df, 'genre', 10, 'genre_top10')            # 상위 10 genre
df = create_top_n_column(df, 'country', 5, 'country_top5')          # 상위 5 country
df = create_top_n_column(df, 'company', 10, 'company_top10')        # 상위 10 company


# Remove existing column
df.drop(columns=[
    'director', 'writer', 'star', 
    'genre', 'country', 'company'
], inplace=True)

# is_hit: If gross is more than twice the budget, it is considered a box office success; 0 (failure) or 1 (hit)
df['is_hit'] = (df['gross'] > 2 * df['budget']).astype(int)

df.to_csv("data/feature-engineered/movies_feature_engineered_basic.csv", index=False)

print("Saved Dataset: data/feature-engineered/movies_feature_engineered_basic.csv")