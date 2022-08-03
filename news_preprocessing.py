import pandas as pd
from text_preprocessing import preprocess

print("Loading dataset...")
true = pd.read_csv("True.csv")
true['label'] = 1
false = pd.read_csv("Fake.csv")
false['label'] = 0
df = pd.concat([true, false], ignore_index=True)

print("Preprocessing data...")
df['title'] = df['title'].apply(lambda i: preprocess(i))
df = df.sample(frac=1).reset_index(drop=True)
df[['title', 'label']].to_csv('preparedData.csv', index=False)
