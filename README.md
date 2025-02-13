# Asthma Disease Prediction Using Big Data and Apache Spark

## Project Overview
This project utilizes **Apache Spark** to preprocess and analyze a large-scale **asthma disease dataset** from Kaggle, containing health data for **2,392 patients**. By examining clinical, environmental, and lifestyle factors, we aim to identify key contributors to asthma and predict its likelihood. This project showcases **big data processing with Spark** and **machine learning techniques**, contributing to a deeper understanding of asthma risks and patterns.

## Dataset
The project is based on the **Asthma Disease dataset from Kaggle**, which includes a diverse set of health indicators for 2,392 patients. This dataset encompasses critical features for assessing asthma risk factors and patient health profiles.

## Goal
- Identify significant factors influencing asthma risk.
- Utilize big data processing techniques to support healthcare research.

## Technologies Used
- **Apache Spark**: For scalable data processing on a cluster environment.
- **Python**: Used for data analysis and modeling with **pandas, scikit-learn, and seaborn**.
- **Machine Learning Models**: Logistic Regression, Linear Regression, and K-Means Clustering.

## Setup and Configuration
### 1. Virtual Cluster Setup
- Connected to the **Michigan Tech network** via the **F5 Big-IP Edge VPN client**.
- Configured network settings to establish a **Spark cluster** with:
  - `hadoop1` as the **master node**.
  - `hadoop2` as the **worker node**.

### 2. Spark Setup
- Installed **Apache Spark** in `/opt/spark`.
- Configured Spark **master and worker nodes**, with:
  - Master running on **hadoop1**.
  - Workers deployed across both VMs.

### 3. Running the Analysis
- Submitted preprocessing and analysis jobs using:
  ```bash
  /opt/spark/bin/spark-submit --master spark://hadoop1:7077 /opt/asthma.py
  ```

## Data Preprocessing
### 1. Handling Missing Values
- Replaced missing values in critical variables (**Age, BMI, DietQuality, LungFunctionFEV1**) with their median.

### 2. Encoding Categorical Variables
- Applied **label encoding** to categorical features (**Gender, FamilyHistoryAsthma**) for compatibility with machine learning models.

### 3. Scaling
- Normalized continuous variables to a range of **0-1** using **MinMaxScaler** to improve model performance.

### 4. Binning
- Converted continuous features (**Age, BMI, LungFunctionFEV1**) into bins for better interpretability and model optimization.

## Data Splitting
To ensure effective model training and evaluation, we divided the dataset into **training (70%)** and **testing (30%)** sets.

- **Classification Task**: Used for **asthma diagnosis prediction** with Logistic Regression.
- **Regression Task**: Used to predict **lung function (LungFunctionFEV1)** using Linear Regression.

## Statistical Analysis
### 1. Correlation Analysis
- Assessed relationships between features like **BMI, DietQuality, and LungFunctionFEV1**.
- Heatmaps revealed low correlations, suggesting **non-linear models** might be more effective.

### 2. Chi-Square Tests
- Analyzed associations between categorical variables (**Gender, PetAllergy**) and **asthma diagnosis**.
- Identified categorical features with potential predictive value.

### 3. K-Means Clustering
- Clustered patients into **three groups** based on continuous variables.
- Helped segment patient health profiles for personalized asthma management.

## Machine Learning Models
### 1. Logistic Regression (Classification)
- Achieved **94.8% accuracy** in predicting asthma diagnosis.
- Feature scaling and binning enhanced model performance.

### 2. Linear Regression (Regression)
- Built a model to predict **LungFunctionFEV1**.
- Attained an **R² score of 1.0**, indicating high accuracy, but further validation is needed to avoid overfitting.

## Results
- Demonstrated **Apache Spark’s capability** to process large datasets efficiently.
- Achieved **high accuracy** in both classification and regression tasks.
- Statistical analysis provided **insights into asthma risk factors** and patient segmentation.

## Conclusion
This project highlights the **power of big data tools** like Apache Spark in healthcare analytics. By efficiently processing and analyzing large datasets, we identified crucial **asthma risk factors** and built predictive models to enhance early detection. The findings contribute to **improving asthma management and prevention strategies** through **data-driven insights**.

