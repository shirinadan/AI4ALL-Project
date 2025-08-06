import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === Load the cleaned dataset ===
df = pd.read_csv("cleaned_investments_VC.csv")

# === Calculate round depth ===
round_columns = [
    "round_A",
    "round_B",
    "round_C",
    "round_D",
    "round_E",
    "round_F",
    "round_G",
    "round_H",
]
round_weights = [1, 2, 3, 4, 5, 6, 7, 8]
df["round_depth"] = df[round_columns].fillna(0).dot(round_weights)

# === Calculate number of funding types ===
funding_types = [
    "seed",
    "venture",
    "equity_crowdfunding",
    "angel",
    "grant",
    "private_equity",
    "post_ipo_equity",
]
df["num_funding_types"] = df[funding_types].fillna(0).sum(axis=1)

# === Success Score v1: Acquisition + log-normalized funding ===
df["success_score_v1"] = 2 * df["is_acquired"] + np.log1p(
    df["funding_total_usd"]
) / np.log1p(df["funding_total_usd"].max())

# === v2 and v3: We'll skip these as they were distorted (can debug later if needed) ===
# # Score v2: Add round depth (optional rescale needed)
# # Score v3: Add funding type diversity (optional rescale needed)

# === Success Score v4: Pure log funding ===
df["success_score_v4"] = np.log1p(df["funding_total_usd"])

# === Success Score v5: Normalized blend of v1 and v4 ===
# Normalize both to 0–1
v1_norm = (df["success_score_v1"] - df["success_score_v1"].min()) / (
    df["success_score_v1"].max() - df["success_score_v1"].min()
)
v4_norm = (df["success_score_v4"] - df["success_score_v4"].min()) / (
    df["success_score_v4"].max() - df["success_score_v4"].min()
)

# Blend
df["success_score_v5"] = 0.5 * v1_norm + 0.5 * v4_norm

# === Plot all success scores ===
plt.figure(figsize=(16, 10))
score_versions = ["v1", "v4", "v5"]
titles = {
    "v1": "Success Score V1 (Acquisition + Normalized Funding)",
    "v4": "Success Score V4 (Log Funding Only)",
    "v5": "Success Score V5 (Normalized Blend of V1 and V4)",
}

for i, version in enumerate(score_versions, start=1):
    plt.subplot(2, 2, i)
    sns.histplot(df[f"success_score_{version}"], bins=50)
    plt.title(titles[version])
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.grid(True)

plt.tight_layout()
plt.show()

# === Print summary stats ===
print("\n=== Summary Statistics ===\n")
for version in score_versions:
    print(f"Success Score {version.upper()} Stats:")
    print(df[f"success_score_{version}"].describe())
    print("\n")

# === Save updated dataset ===
df.to_csv("cleaned_investments_VC_with_scores.csv", index=False)
print("✅ Saved updated dataset to cleaned_investments_VC_with_scores.csv")
