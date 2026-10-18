"""Create exploratory data analysis charts. Run: python eda.py"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "customer_data_cleaned.csv")
out = ROOT / "reports" / "figures"
out.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 5))
df["spending"].dropna().hist(bins=25, edgecolor="black")
plt.title("Distribution of Customer Spending")
plt.xlabel("Spending"); plt.ylabel("Count"); plt.tight_layout()
plt.savefig(out / "spending_distribution.png", dpi=160); plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(df["income"], df["spending"], alpha=.65)
plt.title("Income vs Spending"); plt.xlabel("Income"); plt.ylabel("Spending")
plt.tight_layout(); plt.savefig(out / "income_vs_spending.png", dpi=160); plt.close()

df.groupby("education")["spending"].mean().sort_values().plot(kind="bar", figsize=(8, 5))
plt.title("Average Spending by Education"); plt.xlabel("Education"); plt.ylabel("Average spending")
plt.xticks(rotation=30, ha="right"); plt.tight_layout()
plt.savefig(out / "average_spending_by_education.png", dpi=160); plt.close()
print(f"Charts saved in {out}")
