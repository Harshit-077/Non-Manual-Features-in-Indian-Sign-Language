import pandas as pd
from datasets import load_dataset
from pathlib import Path

TOKEN = "your_hf_token_here"

# ── iSign CSV ─────────────────────────────────────────────
print("=" * 50)
print("iSign CSV")
print("=" * 50)

df = pd.read_csv("isign_workspace/metadata/iSign_v1.1.csv")

print(f"Total sentences : {len(df)}")
print(f"Unique videos   : {df['uid'].str.split('-').str[0].nunique()}")
print(f"Avg text length : {df['text'].str.len().mean():.1f} chars")
df['text_len'] = df['text'].str.len()
print(f"\nShortest texts:")
print(df.nsmallest(5, 'text_len')['text'].values)
print(f"\nLongest texts:")
print(df.nlargest(3, 'text_len')['text'].values)


# ── CISLR ─────────────────────────────────────────────────
print("\n" + "=" * 50)
print("CISLR")
print("=" * 50)

cislr = load_dataset("IIT-K/CISLR", token=TOKEN)["test"]

cislr_df = pd.DataFrame(cislr)

print(f"Total signs     : {len(cislr_df)}")
print(f"Unique glosses  : {cislr_df['gloss'].nunique()}")
print(f"Unique categories: {cislr_df['category'].nunique()}")
print(f"\nCategories:")
print(cislr_df['category'].value_counts().to_string())
print(f"\nAvg duration    : {cislr_df['duration'].mean():.1f}s")
print(f"Min duration    : {cislr_df['duration'].min():.1f}s")
print(f"Max duration    : {cislr_df['duration'].max():.1f}s")
print(f"\nSample glosses per category:")
for cat, group in cislr_df.groupby("category"):
    samples = group['gloss'].sample(min(3, len(group)), random_state=42).tolist()
    print(f"  {cat:20s}: {samples}")
