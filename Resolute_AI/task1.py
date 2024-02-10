import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px

def main():
    st.title('K-Means Clustering with Streamlit')

    # Load data
    @st.cache
    def load_data():
        df = pd.read_excel('C:\Users\Admin\Desktop\Datasets\Resolute_AI\Resolute_AI\Data\Task1and2\train.xlsx')
        df = df.drop(columns=['target'])
        return df

    df = load_data()

    # Display data
    st.subheader('Data')
    st.write(df.head())

    # KMeans clustering
    st.subheader('KMeans Clustering')

    k = 180  # Fixed number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df)
    silhouette_avg = silhouette_score(df, kmeans.labels_)
    st.write(f"Silhouette Score for {k} clusters:", silhouette_avg)

    df['Cluster'] = kmeans.labels_

    # Show dataframe after clustering
    st.subheader('Data after Clustering')
    st.write(df.head())

    # PCA Visualization
    st.subheader('PCA Visualization')

    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(df.drop(columns=['Cluster']))

    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])
    pca_df['Cluster'] = df['Cluster']

    fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='Cluster', opacity=0.7,
                        title='PCA Visualization of Clusters', labels={'Cluster': 'Cluster'})
    fig.update_layout(scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))

    st.plotly_chart(fig)

    # Allow user to input data and predict cluster
    st.subheader('Predict Cluster for New Data')
    new_data = []
    for i in range(18):
        value = st.number_input(f'Input {i+1}', value=0.0)
        new_data.append(value)
    
    if st.button('Predict'):
        new_data.append(0)  # Adding a placeholder value for the 'Cluster' feature
        new_data_array = np.array(new_data).reshape(1, -1)
        predicted_cluster = kmeans.predict(new_data_array)
        st.write(f'The predicted cluster for the input data is: {predicted_cluster[0]}')

if __name__ == '__main__':
    main()
