SUMMARY:
Looking to predict Walmart's sales accurately? We use smart methods to understand what customers want. By diving deep into data, we find important trends and patterns that affect what people buy. Our models are like finely-tuned machines, giving reliable results and adjusting to changes in the market. We're always looking for ways to make our predictions even better, so you stay ahead of the game. And we don't stop at predictions. We give practical advice tailored to Walmart's challenges, helping you make real-world decisions that work. In short, we help Walmart understand what customers want, predict sales accurately, and give practical advice for success.

STEPS OF MODEL IMPLEMENTATION: 
1. MOUNT DRIVE
2. IMPORT LIBRARIES
3. LOAD DATASET
4. PRINT FIRST 5 ROWS
5. EDA: understands the data, identifies patterns, spot anomalies, and formulate hypotheses that can be tested further.
6. DATA PRE-PROCESSING
	A. HANDLING MISSING VALUES
	B. FEATURE ENGINEERING

7. DATA VISUALIZATION : It is the graphical representation of data. It involves creating visual elements like charts, graphs, and maps to communicate insights from the data effectively.
	A. Best season - summer
	B. TOTAL SALES IN EACH YR - 2011
	C. TOTAL SALES IN EACH MONTH - july
	D. TOTAL SALES IN EACH WEEK - week 51
	E. HEAT MAP
		a. 1's along the diagonal and 0's elsewhere, it indicates that each variable is perfectly correlated with itself i.e Correlation = 1
		b. 0's elsewhere indicate that there are no strong correlations (greater than 0.8) between any pair of variables in the dataset.

8. TYPE CASTING
	A. To make sure no changes are made in the original Data.
	B. Thus the code is copied and assigned to df1
	C. Features like Store, holiday flag and week are converted (explicitly) from numerical data to categorical data for better analysis and modelling

9. DATA TRANSFORMATION: 
	A. Splitting the data into Numerical Features and Categorical Features 
	B. Numerical Features : ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment'] 
	C. Categorical Features: ['Store', 'Holiday_Flag', 'week']

10. DETECTING AND REMOVING OUTLIERS
	A.  OUTLIERS:
		a. Outlier detection is a method used to find unusual or abnormal data points in a set of information. In data, outliers are points that deviate significantly from the majority, and detecting them helps identify unusual patterns or errors in the information.
		b. Total outliers: 675 when threshold = 3
		c. 3 is standard deviation from mean is a common approach holding about 99.7% of data under a normal distribution
	B. Z-SCORE :
 		a. z-scores is to standardise data and compare individual data points to the overall dataset. It helps in identifying how far a particular data point is from the mean of the dataset, measured in terms of standard deviations.


11. SPLITTING THE DATA INTO TRAINING AND TESTING DATA
	Features - independent variable (excluding Weekly_Sales)
	Target variable - Weekly_Sales
	X_train: Training data - features
	X_test: Testing Data - features
	y_train: Training data on target variables
	y_test: Testing data on target variables


12. StandardScaler : helps prevent larger scales from dominating the training process
    Binary Encoder: 
	A. encoding categorical features into binary format; 
	B. REDUCES DIMENSIONALITY
	C. Binary encoding simplifies the dataset by using fewer features than one-hot encoding, which saves memory, particularly for categories with many unique values.

13. HYPERPARAMETER TUNING  ( GRID SEARCH + RANDOMISED SEARCH )
	A. Hyperparameter tuning is the process of selecting the optimal values for a machine learning model’s hyperparameters. Hyperparameters are settings that control the learning process of the model, such as the learning rate, the number of neurons in a neural network, or the kernel size in a support vector machine. The goal of hyperparameter tuning is to find the values that lead to the best performance on a given task.
	B. OBSERVATION BETWEEN 2 METHODS : since the dataset isn’t that large because of which the number hyperparameters are also not in large numbers i.e not computationally expensive. Thus the accuracy of models is better with grid search over randomised search.

14. CROSS VALIDATION
	A. Cross-validation is a technique for evaluating ML models by training several ML models on subsets of the available input data and evaluating them on the complementary subset of the data. Use cross-validation to detect overfitting 
	B. Estimator: an object that represents a predictive model


15. MODELS
	A. Supervised Learning
		a. Linear Regression predicts unknown data by using known related data. It establishes a global estimator by identifying linear relationships between variables. It fits a line to the data for making predictions.
		b. Polynomial Regression captures non-linear relationships by fitting non-linear regression lines.
		c. Ridge Regression estimates coefficients in multiple-regression models when the independent variables are highly correlated.
		d. Lasso Regression improves predictive accuracy by pulling data towards a central point, often represented by the mean.
		e. Decision Tree Regressor predicts continuous target variables by dividing feature variables into zones with individual predictions. Overfitting is a phenomenon to watch out for.
		f. Random Forest Regressor enhances predictive accuracy and controls overfitting by using multiple decision tree regression on subsets of the dataset.
		g. KNN Regressor provides local estimation based on the similarity of data points in their neighbourhood.
		h. XGB Regressor stands for Extreme Gradient Boosting. It offers scalable, distributed gradient-boosted decision tree (GBDT) machine learning. It is a leading library for solving regression, classification, and ranking problems.
	B. Unsupervised Learning
		a. K Means Clustering is a group of unlabeled datasets into pre-defined clusters based on their similarity.
		b. Hierarchical Clustering organises groups where objects within a group share similarities and differ from other groups. It can be visualised using dendrograms in a hierarchical tree.
	C. Deep Learning
		a. Multi-Layer Perceptron (MLP) is a neural network with interconnected layers. It uses the BackPropagation algorithm for training.

16. EVALUATION METRICS
	A. SUPERVISED LEARNING : REGRESSION
		a. Mean Absolute Error (MAE)
		b. Mean Squared Error (MSE)
		c. Root Mean Squared Error (RMSE)
		d. R Squared (R2)
	B. UNSUPERVISED LEARNING
		a. Silhouette score
		b. index
	C. DEEP LEARNING
		a. Mean Absolute Error (MAE)
		b. Mean Squared Error (MSE)
		c. Root Mean Squared Error (RMSE)
		d. R Squared (R2)

17. VISUALIZATION: DISTRIBUTION PLOT 
