# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Dataset
df = pd.read_csv("C:/Users/Surji/OneDrive/Desktop/ca2 project/new python .csv")
print("Dataset Head:\n", df.head())

# 3. Dataset Info
print("\nINFO:\n")
print(df.info())

# 4. Check Missing Values
print("\nMissing Values:\n", df.isnull().sum())
df_cleaned = df.fillna(0)
print(df_cleaned)

# 5. Summary Statistics
print("\nSummary Statistics:\n", df.describe())

# 6. Dataset Shape and Columns
print("\nDataset Shape:", df.shape)
print("Columns:", df.columns.tolist())

# 7. Unique Years/States
print("\nUnique Years:", df['YEAR'].unique())
print("Unique Subdivisions:", df['SD_Name'].unique())

# 8. Group-wise Annual Rainfall
annual_rainfall = df.groupby('YEAR')['ANNUAL'].sum()
print("\nTotal Annual Rainfall by Year:\n", annual_rainfall.head())

# 9. Correlation Matrix
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
corr = df[numerical_cols].corr()
print("\nCorrelation Matrix:\n", corr)

# 10. Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Rainfall Features")
plt.tight_layout()
plt.show()

# 11. Line Plot: Annual Rainfall over Years
plt.figure(figsize=(10, 6))
annual_rainfall.plot(marker='o', color='blue')
plt.title("Total Annual Rainfall Over Years")
plt.xlabel("Year")
plt.ylabel("Total Rainfall (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 12. Subdivision-wise Average Rainfall
avg_by_region = df.groupby('SD_Name')['ANNUAL'].mean().sort_values(ascending=False).reset_index()
print("\nAverage Rainfall by Subdivision:\n", avg_by_region)

# Bar Plot
plt.figure(figsize=(12, 8))
sns.barplot(data=avg_by_region, x='ANNUAL', y='SD_Name', hue='SD_Name', palette='Set3', legend=False)
plt.title("Average Annual Rainfall by Subdivision")
plt.xlabel("Average Rainfall (mm)")
plt.ylabel("Subdivision")
plt.tight_layout()
plt.show()

# 13. Rainfall in a Selected State Over Time (e.g., PUNJAB)
punjab_data = df[df['SD_Name'].str.upper().str.contains("PUNJAB")]
plt.figure(figsize=(10, 5))
plt.plot(punjab_data['YEAR'], punjab_data['ANNUAL'], marker='o', linestyle='-', color='green')
plt.title("Annual Rainfall in Punjab Over Time")
plt.xlabel("Year")
plt.ylabel("Annual Rainfall (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 14. Monthly Rainfall Distribution (Boxplot)
monthly_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
monthly_df = df[monthly_cols]

plt.figure(figsize=(12, 6))
sns.boxplot(data=monthly_df, palette='Set2')
plt.title("Monthly Rainfall Distribution Across India")
plt.xlabel("Month")
plt.ylabel("Rainfall (mm)")
plt.tight_layout()
plt.show()

# 15. Seasonal Rainfall Comparison
seasonal_cols = ['JAN-FEB', 'Mar-May', 'Jun-Sep', 'Oct-Dec']
seasonal_means = df[seasonal_cols].mean()
print("\nSeasonal Rainfall Mean (All India):\n", seasonal_means)

# Pie Chart
plt.figure(figsize=(7, 7))
colors = sns.color_palette("Set3")
plt.pie(seasonal_means, labels=seasonal_means.index, autopct='%1.1f%%', colors=colors)
plt.title("Seasonal Contribution to Total Rainfall")
plt.tight_layout()
plt.show()

# 16. Scatter Plot: JUN-SEP vs ANNUAL
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Jun-Sep', y='ANNUAL', color='purple', alpha=0.6)
plt.title("JUN-SEP Rainfall vs Annual Rainfall")
plt.xlabel("JUN-SEP Rainfall (mm)")
plt.ylabel("Annual Rainfall (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 17. Regression Plot: JUN-SEP vs ANNUAL
plt.figure(figsize=(8, 6))
sns.regplot(data=df, x='Jun-Sep', y='ANNUAL', color='darkblue', alpha=0.4)
plt.title("Regression: JUN-SEP vs Annual Rainfall")
plt.xlabel("JUN-SEP Rainfall (mm)")
plt.ylabel("Annual Rainfall (mm)")
plt.tight_layout()
plt.show()

# 18. Regression Plot: MAR-MAY vs ANNUAL
plt.figure(figsize=(8, 6))
sns.regplot(data=df, x='Mar-May', y='ANNUAL', color='orange', scatter_kws={"alpha":0.5})
plt.title("Regression: MAR-MAY vs Annual Rainfall")
plt.xlabel("MAR-MAY Rainfall (mm)")
plt.ylabel("Annual Rainfall (mm)")
plt.tight_layout()
plt.show()

# 19. Scatter: Year vs Annual Rainfall for All India
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='YEAR', y='ANNUAL', alpha=0.6)
plt.title("Annual Rainfall Trend Across India")
plt.xlabel("Year")
plt.ylabel("Annual Rainfall (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 20. Violin Plot: Monthly Rainfall Distribution
# Reshape monthly columns to long format
monthly_df = df[monthly_cols].melt(var_name='Month', value_name='Rainfall')

# Violin Plot
plt.figure(figsize=(12, 6))
sns.violinplot(x='Month', y='Rainfall', data=monthly_df, palette='Pastel2')
plt.title("Violin Plot of Monthly Rainfall Distribution Across India")
plt.xlabel("Month")
plt.ylabel("Rainfall (mm)")
plt.tight_layout()
plt.show()

