# INVENTORY DEMAND FORECASTING

## OVERVIEW
Looking to predict Walmart's sales accurately? We use smart methods to understand what customers want. By diving deep into data, we find important trends and patterns that affect what people buy. Our models are like finely-tuned machines, giving reliable results and adjusting to changes in the market. We're always looking for ways to make our predictions even better, so you stay ahead of the game. And we don't stop at predictions. We give practical advice tailored to Walmart's challenges, helping you make real-world decisions that work. In short, we help Walmart understand what customers want, predict sales accurately, and give practical advice for success.

## STEPS OF MODEL IMPLEMENTATION:
1. **MOUNT DRIVE**  
2. **IMPORT LIBRARIES**  
3. **LOAD DATASET**  
4. **PRINT FIRST 5 ROWS**  
5. **EDA**:  
   Understands the data, identifies patterns, spots anomalies, and formulates hypotheses that can be tested further.  

6. **DATA PRE-PROCESSING**  
   - **A. HANDLING MISSING VALUES**  
   - **B. FEATURE ENGINEERING**  

7. **DATA VISUALIZATION**  
   It is the graphical representation of data. It involves creating visual elements like charts, graphs, and maps to communicate insights from the data effectively.  
   - **A. Best season** - summer  
   - **B. TOTAL SALES IN EACH YEAR** - 2011  
   - **C. TOTAL SALES IN EACH MONTH** - July  
   - **D. TOTAL SALES IN EACH WEEK** - Week 51  
   - **E. HEAT MAP**  
     - **a.** 1's along the diagonal and 0's elsewhere, it indicates that each variable is perfectly correlated with itself i.e., Correlation = 1  
     - **b.** 0's elsewhere indicate that there are no strong correlations (greater than 0.8) between any pair of variables in the dataset.  

8. **TYPE CASTING**  
   - **A.** To make sure no changes are made in the original Data.  
   - **B.** Thus, the code is copied and assigned to `df1`.  
   - **C.** Features like `Store`, `Holiday_Flag`, and `Week` are converted (explicitly) from numerical data to categorical data for better analysis and modeling.  

9. **DATA TRANSFORMATION**  
   - **A. Splitting the data into Numerical Features and Categorical Features**  
   - **B. Numerical Features**:  
     `['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']`  
   - **C. Categorical Features**:  
     `['Store', 'Holiday_Flag', 'Week']`  

10. **DETECTING AND REMOVING OUTLIERS**  
    - **A. OUTLIERS:**  
      - **a.** Outlier detection is a method used to find unusual or abnormal data points in a set of information. In data, outliers are points that deviate significantly from the majority, and detecting them helps identify unusual patterns or errors in the information.  
      - **b.** Total outliers: 675 when threshold = 3  
      - **c.** 3 is standard deviation from mean is a common approach holding about 99.7% of data under a normal distribution  
    - **B. Z-SCORE:**  
      - **a.** Z-scores standardize data and compare individual data points to the overall dataset. It helps in identifying how far a particular data point is from the mean of the dataset, measured in terms of standard deviations.  

11. **SPLITTING THE DATA INTO TRAINING AND TESTING DATA**  
    - Features - independent variable (excluding `Weekly_Sales`)  
    - Target variable - `Weekly_Sales`  
    - `X_train`: Training data - features  
    - `X_test`: Testing Data - features  
    - `y_train`: Training data on target variables  
    - `y_test`: Testing data on target variables  

12. **StandardScaler**: Helps prevent larger scales from dominating the training process.  
    **Binary Encoder:**  
    - **A.** Encoding categorical features into binary format.  
    - **B.** REDUCES DIMENSIONALITY.  
    - **C.** Binary encoding simplifies the dataset by using fewer features than one-hot encoding, which saves memory, particularly for categories with many unique values.  

13. **HYPERPARAMETER TUNING (GRID SEARCH + RANDOMIZED SEARCH)**  
    - **A.** Hyperparameter tuning is the process of selecting the optimal values for a machine learning model’s hyperparameters. Hyperparameters are settings that control the learning process of the model, such as the learning rate, the number of neurons in a neural network, or the kernel size in a support vector machine. The goal of hyperparameter tuning is to find the values that lead to the best performance on a given task.  
    - **B. OBSERVATION BETWEEN 2 METHODS:**  
      Since the dataset isn’t that large, the number of hyperparameters is also not in large numbers i.e., not computationally expensive. Thus, the accuracy of models is better with grid search over randomized search.  

14. **CROSS VALIDATION**  
    - **A.** Cross-validation is a technique for evaluating ML models by training several ML models on subsets of the available input data and evaluating them on the complementary subset of the data. Use cross-validation to detect overfitting.  
    - **B. Estimator:** An object that represents a predictive model.  

15. **MODELS**  
    - **A. Supervised Learning**  
      - **a.** Linear Regression  
      - **b.** Polynomial Regression  
      - **c.** Ridge Regression  
      - **d.** Lasso Regression  
      - **e.** Decision Tree Regressor  
      - **f.** Random Forest Regressor  
      - **g.** KNN Regressor  
      - **h.** XGB Regressor  
    - **B. Unsupervised Learning**  
      - **a.** K Means Clustering  
      - **b.** Hierarchical Clustering  
    - **C. Deep Learning**  
      - **a.** Multi-Layer Perceptron (MLP)  

16. **EVALUATION METRICS**  
    - **A. SUPERVISED LEARNING: REGRESSION**  
      - Mean Absolute Error (MAE)  
      - Mean Squared Error (MSE)  
      - Root Mean Squared Error (RMSE)  
      - R Squared (R2)  
    - **B. UNSUPERVISED LEARNING**  
      - Silhouette Score  
      - Index  
    - **C. DEEP LEARNING**  
      - Mean Absolute Error (MAE)  
      - Mean Squared Error (MSE)  
      - Root Mean Squared Error (RMSE)  
      - R Squared (R2)  

17. **VISUALIZATION: DISTRIBUTION PLOT**  
