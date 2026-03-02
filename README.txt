Mall Customer Segmentation

Overview

This project applies unsupervised machine learning techniques to segment mall customers based on their age, annual income, and spending score. It covers the full pipeline — feature engineering, skewness correction, scaling, dimensionality reduction with PCA and T-SNE, and clustering using multiple algorithms.

This was a group project completed during second year in university.
Group members: Tareq M. Hab-El-Rumman and Gina R. Maayah.
---------------------------------------

Dataset

The Mall Customers dataset contains basic demographic and behavioral data for 200 mall customers, including Gender, Age, Annual Income (k$), and Spending Score (1-100).

To run the notebook locally, Keep Mall_Customers.csv inside the folder named data/, just like you downloaded it from the repository.
---------------------------------------

What We Did

1. Data Cleaning

. Dropped the CustomerID column as it has no predictive value
. Verified there were no missing values or duplicate rows
. Encoded Gender using LabelEncoder

2. Feature Engineering

New features were created to capture more meaningful patterns:

. Income_per_age — annual income divided by age
. Spending_per_Age — spending score divided by age
. Spending_per_Income — spending score divided by annual income

3. Skewness Correction

. Applied log1p transformation to all numeric features (excluding Gender) to reduce skewness
. Visualized distributions before and after transformation using histograms

4. Imputation

. Used SimpleImputer with mean strategy for all numeric features
. Used SimpleImputer with most_frequent strategy for Gender

5. Scaling

. Applied StandardScaler to all features before dimensionality reduction and clustering

6. Dimensionality Reduction

. PCA retaining 95% variance — used to reduce dimensions before clustering
. 2D PCA — used for visualization before and after scaling
. T-SNE — applied before and after PCA for visual comparison of cluster structure

7. Clustering

Three clustering algorithms were compared across K values from 2 to 9:

. KMeans (random init)
. KMeans++ (k-means++ init)
. MiniBatchKMeans

Best K was selected based on the highest silhouette score for each method.\
DBSCAN was also applied with a grid search over eps and min_samples to find the best configuration.
---------------------------------------

Results

 Model                Best K    Silhouette
. KMeans (random)     8         highest among K=2–9
. KMeans++            9         highest among K=2–9
. MiniBatchKMeans     7         highest among K=2–9

KMeans++ produced the most compact and well-separated clusters.\
DBSCAN was able to detect outliers and handled irregular cluster shapes better than KMeans, though both provide useful insights depending on the analysis focus.
---------------------------------------

Visualizations

. Correlation heatmap of numeric features

. Distribution histograms before and after skewness correction

. Box plots for each feature

. PCA explained variance plot

. 2D PCA scatter plots before and after scaling

. T-SNE scatter plots before and after PCA

. Inertia and silhouette score vs. K for all three KMeans variants

. Silhouette plots for K = 7, 8, 9

. DBSCAN clustering plot (grey points = outliers)

. Side-by-side comparison of all three clustering methods on PCA-reduced data