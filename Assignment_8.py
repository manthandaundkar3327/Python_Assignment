import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. LOAD A SIMPLE SAMPLE DATASET (Penguins)
df  = sns.load_dataset("penguins")
print(df.head(50))

# Drop missing values to keep things simple
df = df.dropna()

print(df.head(10))
print(df.info())

# --------------------------------------------------
# STEP 1: CREATE A SCATTER PLOT

plt.figure(figsize=(7, 5))

# Plot Bill Length vs. Bill Depth
sns.scatterplot(
    data=df,
    x="bill_length_mm",
    y="bill_depth_mm",
    hue="species"   # Colors points by penguin species
)

plt.title("Scatter Plot: Bill Length vs. Bill Depth")
plt.xlabel("Bill Length (mm)")
plt.ylabel("Bill Depth (mm)")

plt.show()

# --------------------------------------------------
# STEP 2: CREATE A CORRELATION HEATMAP-----

plt.figure(figsize=(6, 5))

# Calculate numerical correlation between numbers only
numeric_data = df.select_dtypes(include="number")
correlation_matrix = numeric_data.corr()

# Draw the heatmap
sns.heatmap(
    correlation_matrix,
    annot=True,       # Shows the correlation number inside each box
    cmap="coolwarm",  # Red for positive correlation, Blue for negative
    fmt=".2f"         # Rounds numbers to 2 decimal places
)

plt.title("Correlation Heatmap")

plt.show()