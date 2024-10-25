# Asthma-Disease-Prediction-Using-Big-Data-Spark
Project Overview:

This project leverages Apache Spark to preprocess and analyze a large-scale asthma disease dataset from Kaggle, containing health data for 2,392 patients. By analyzing clinical, environmental, and lifestyle factors, our goal is to uncover key contributors to asthma and predict its likelihood. This project demonstrates big data handling with Spark and machine learning techniques, enhancing our understanding of asthma risks and patterns.
 
Dataset:

The project uses the Asthma Disease dataset from Kaggle, which contains extensive health data from 2,392 patients. This dataset includes a variety of features crucial for analyzing asthma risk factors and patient health profiles

Goal: 

Identify significant factors influencing asthma risk and apply big data processing techniques to support healthcare research.

Technologies Used:

Apache Spark: For data processing and scaling on a cluster environment.
Python: For data analysis and modeling with libraries like pandas, scikit-learn, and seaborn.
Machine Learning Models: Logistic Regression, Linear Regression, K-Means Clustering.

Setup and Configuration:

1.Virtual Cluster Setup:

Accessed the Michigan Tech network via the F5 Big-IP Edge VPN client.
Configured network settings to establish a Spark cluster on VMs (hadoop1 as the master and hadoop2 as the worker node).

2.Spark Setup:

Installed Spark in /opt/spark.
Configured Spark master and worker nodes, starting the master on hadoop1 and workers on both VMs.

3.Running the Analysis:

Submitted preprocessing and analysis jobs using:

/opt/spark/bin/spark-submit --master spark://hadoop1:7077 /opt/asthma.py

Data Preprocessing:

1.Handling Missing Values: Filled missing values in key variables like Age, BMI, DietQuality, and lung function measures to improve model performance.

2.Encoding Categorical Variables: Used label encoding on features such as Gender and FamilyHistoryAsthma for model compatibility.

3.Scaling: Scaled continuous variables to a range of 0-1 using MinMaxScaler, normalizing values for balanced input.

4.Binning: Applied binning to continuous features (Age, BMI, LungFunctionFEV1) to enhance interpretability.

Data Splitting:

To ensure effective model training and evaluation, the dataset was divided into training and testing sets:

1.Classification Task: For asthma diagnosis prediction, the data was split with 70% allocated for training and 30% for testing. This split enabled the logistic regression model to be trained effectively and evaluated independently.

2.Regression Task: For predicting lung function (LungFunctionFEV1), the same 70/30 split was used to train and test the linear regression model, allowing for consistent evaluation across both tasks.

By splitting the dataset appropriately, the project achieved balanced training and testing data, providing an accurate measure of model performance and preventing

Statistical Analysis:

1.Correlation Analysis: Examined correlations among features like BMI, DietQuality, and lung function measures, using heatmaps to visualize relationships. The low correlation levels suggested complex interactions, indicating that non-linear models may be more appropriate for analysis.

2.Chi-Square Tests: Explored associations between categorical variables (e.g., Gender, PetAllergy) and asthma diagnosis. These tests provided insights into relationships among features, helping identify categorical factors with potential predictive value for asthma.

3.K-Means Clustering: Implemented K-Means clustering to segment patients into three clusters, based on continuous variables. These clusters reflect common health profiles, supporting tailored asthma management.

Modeling:

1.Logistic Regression: Achieved ~94.8% accuracy for asthma diagnosis by training a Logistic Regression model, optimizing it with binning and scaling of features. This model provided a solid classification accuracy, confirming the effectiveness of preprocessing.

2.Linear Regression: Built a linear regression model to predict lung function (LungFunctionFEV1) based on continuous health factors. The model achieved an R² of 1.0, though additional testing is needed to ensure robustness and prevent overfitting.

Results:

The project successfully demonstrated Spark’s capability to handle large datasets efficiently. Models for classification and regression were implemented with high accuracy, and statistical analysis provided insights into asthma risk factors and patient clusters.

Conclusion:

This project highlights the utility of big data tools like Spark in healthcare data analysis, identifying valuable insights into asthma risk factors. By efficiently processing data and building predictive models, this analysis contributes to understanding health patterns and potentially enhancing asthma prevention strategies.
