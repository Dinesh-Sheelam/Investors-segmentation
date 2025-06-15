import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the final dataset
df = pd.read_csv("customer_segments.csv")

st.title("📊 Customer Segmentation for Investors")
st.markdown("""
This app shows how we grouped customers into **4 types** based on how they invest money.  
Even if you’re new to finance, don’t worry! Each group behaves differently, just like students have different hobbies.

**Why do this?**  
So banks and apps can give better advice — like what to invest in — to different types of customers.
""")

# Show dataset overview
st.subheader("🔍 Raw Data Preview")
st.dataframe(df.head())

# Show explanation of segments
st.subheader("🧠 What Do These Segments Mean?")
st.markdown("""
- **Passive Investor** – Invests rarely and simply  
- **Active Trader** – Buys and sells a lot  
- **Diversified Wealth Builder** – Invests in many areas with bigger amounts  
- **Newbie / Infrequent** – Just getting started  
""")

# Plot PCA for cluster visualization
st.subheader("🎨 See the Customer Segments")

# Select features to cluster on
features = ['num_trades', 'total_volume', 'avg_trade_value', 
            'num_assets', 'asset_types', 'sector_diversity', 'investment_span_days']

X_scaled = StandardScaler().fit_transform(df[features])
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)
df['PC1'] = pca_components[:, 0]
df['PC2'] = pca_components[:, 1]

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='segment', palette='Set2', s=60, ax=ax)
plt.title("Customers Grouped by Behavior")
st.pyplot(fig)

# Let user select a segment
st.subheader("🔍 Explore by Segment")
selected = st.selectbox("Choose a segment:", df['segment'].unique())
st.write(f"Showing people in: **{selected}**")

segment_data = df[df['segment'] == selected]
st.dataframe(segment_data[['customerID'] + features].head(10))

# Stats
st.markdown("### 📈 Average Stats for This Segment:")
st.write(segment_data[features].mean().round(2))
