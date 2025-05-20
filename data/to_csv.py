import pandas as pd

# raw CSV URL
url = 'https://raw.githubusercontent.com/ThomasFurtado/Movies-Dataset-IMDb/main/movies.csv'

df = pd.read_csv(url)
df.to_csv('data/raw/movies_original_dataset.csv', index=False, encoding='utf-8-sig') 

print("Saved raw file into data/raw/movies_original_dataset.csv file")
