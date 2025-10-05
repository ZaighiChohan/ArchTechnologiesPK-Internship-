# Step 1: Load dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("Mall_Customers.csv")

print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())

# Check missing values
print("Missing values:\n", df.isnull().sum())

# Drop duplicates
df = df.drop_duplicates()

print("Shape after cleaning:", df.shape)



# Encode Gender as 0/1
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  
# Male = 1, Female = 0 (check with le.classes_)

print(df[['CustomerID','Gender']].head())

# Step 2: EDA
# Gender distribution
sns.countplot(x="Gender", data=df, palette="Set2")
plt.title("Gender Distribution")
plt.show()

# Age distribution
sns.histplot(df["Age"], bins=20, kde=True, color="skyblue")
plt.title("Age Distribution")
plt.show()

# Annual Income distribution
sns.histplot(df["Annual Income (k$)"], bins=20, kde=True, color="green")
plt.title("Annual Income Distribution")
plt.show()

# Spending Score distribution
sns.histplot(df["Spending Score (1-100)"], bins=20, kde=True, color="orange")
plt.title("Spending Score Distribution")
plt.show()

# Income vs Spending Score scatterplot
sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", data=df, hue="Gender", palette="Set1")
plt.title("Annual Income vs Spending Score")
plt.show()

# Step 3: Preprocessing for Clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Select features (Annual Income + Spending Score + Age)
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Find optimal number of clusters
inertia = []
sil_scores = []
K = range(2,11)

for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

# Elbow & Silhouette plots
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(K, inertia, 'bo-')
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.subplot(1,2,2)
plt.plot(K, sil_scores, 'ro-')
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")

plt.tight_layout()
plt.show()

# ==== Fit final KMeans with k=6 ====
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Use the same X and X_scaled from earlier:
# X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
# X_scaled = scaler.fit_transform(X)

k_opt = 6
kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = labels

# ==== Visualize clusters in 2D (Income vs Spending) ====
plt.figure(figsize=(9,7))
sns.scatterplot(
    x=df['Annual Income (k$)'],
    y=df['Spending Score (1-100)'],
    hue=df['Cluster'],
    palette='Set2',
    s=80,
    edgecolor='white'
)
plt.title("Customer Segments (k=6) Income vs Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

# ==== Optional: visualize with Age vs Spending ====
plt.figure(figsize=(9,7))
sns.scatterplot(
    x=df['Age'], y=df['Spending Score (1-100)'],
    hue=df['Cluster'], palette='Set2', s=80, edgecolor='white'
)
plt.title("Customer Segments (k=6) Age vs Spending Score")
plt.tight_layout()
plt.show()

# ==== Cluster centers (in original units) ====
# Inverse-transform the centers from scaled space back to original
centers_scaled = kmeans.cluster_centers_
centers = pd.DataFrame(
    scaler.inverse_transform(centers_scaled),
    columns=X.columns
)
centers['Cluster'] = range(k_opt)
print("Cluster Centers (original units):")
print(centers)

# ==== Profile each cluster ====
profile = (
    df.groupby('Cluster')[['Age','Annual Income (k$)','Spending Score (1-100)']]
      .agg(['count','mean','median','min','max'])
      .round(2)
)
print(profile)

# Quick, human-readable labels (rule-of-thumb based on center values)
def label_segment(row):
    income = row['Annual Income (k$)']
    spend  = row['Spending Score (1-100)']
    age    = row['Age']
    if spend > 70 and income > 70:
        return "Premium Big Spenders"
    if spend > 70 and income <= 70:
        return "Value Big Spenders"
    if 40 < spend <= 70 and income > 70:
        return "Affluent Mid Spenders"
    if spend <= 40 and income > 70:
        return "High Income, Low Spend"
    if spend <= 40 and income <= 70:
        return "Budget-Conscious"
    return "Mid-Market"

centers['Segment'] = centers.apply(label_segment, axis=1)
print(centers[['Cluster','Age','Annual Income (k$)','Spending Score (1-100)','Segment']])

# Attach segment names back to each customer
seg_map = centers.set_index('Cluster')['Segment'].to_dict()
df['Segment'] = df['Cluster'].map(seg_map)

# ==== Example segment counts ====
seg_counts = df['Segment'].value_counts()
print("\nCustomers per Segment:")
print(seg_counts)
