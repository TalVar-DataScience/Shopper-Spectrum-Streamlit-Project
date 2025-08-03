import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load models and data
try:
    kmeans = pickle.load(open('kmeans_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    rfm_data = pd.read_pickle('rfm_data.pkl')
    similarity_df = pd.read_pickle('similarity_matrix.pkl')
    product_data = pd.read_pickle('product_data.pkl')
except FileNotFoundError:
    st.error("Required model/data files not found. Please run the Jupyter Notebook first.")
    st.stop()

# Function to get product recommendations
def get_recommendations(product_code, similarity_df, product_data, top_n=5):
    if product_code not in similarity_df.index:
        return []
    similar_scores = similarity_df[product_code].sort_values(ascending=False)[1:top_n+1]
    product_names = product_data.set_index('StockCode')['Description'].to_dict()
    recommendations = [(product_names.get(code, 'Unknown'), score) for code, score in similar_scores.items()]
    return recommendations

# Function to predict cluster
def predict_cluster(recency, frequency, monetary, scaler, kmeans):
    cluster_labels = {0: 'High-Value', 1: 'Regular', 2: 'Occasional', 3: 'At-Risk'}
    input_data = np.array([[recency, frequency, monetary]])
    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]
    return cluster_labels.get(cluster, 'Unknown')

# Streamlit app
st.title("🛒 Shopper Spectrum: E-Commerce Analytics")

# Sidebar for navigation
st.sidebar.header("Navigation")
module = st.sidebar.radio("Select Module", ["Product Recommendation", "Customer Segmentation"])

# Product Recommendation Module
if module == "Product Recommendation":
    st.header("Product Recommendation")
    product_name = st.text_input("Enter Product Name (StockCode):", "")
    if st.button("Get Recommendations"):
        if product_name:
            recommendations = get_recommendations(product_name, similarity_df, product_data, 5)
            if recommendations:
                st.subheader("Top 5 Recommended Products:")
                for i, (prod, score) in enumerate(recommendations, 1):
                    st.markdown(f"**{i}. {prod}** (Similarity: {score:.2f})")
            else:
                st.warning("Product not found or no recommendations available.")
        else:
            st.warning("Please enter a valid product name.")

# Customer Segmentation Module
else:
    st.header("Customer Segmentation")
    st.write("Enter customer RFM values to predict their segment.")
    
    recency = st.number_input("Recency (days since last purchase)", min_value=0.0, value=30.0)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0.0, value=10.0)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=100.0)
    
    if st.button("Predict Cluster"):
        cluster_label = predict_cluster(recency, frequency, monetary, scaler, kmeans)
        st.subheader(f"Predicted Cluster: **{cluster_label}**")

# Footer
st.markdown("---")
st.markdown("Developed by Nilofer Mubeen | Powered by Streamlit")