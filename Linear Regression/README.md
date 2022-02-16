Predicting Insurance Costs with Multiple Linear Regression
===========================================================

**Description:** This project creates, evaluates, and deploys a Multiple Linear Regression ML model capable of predicting insurance costs based on patient information (age, sex, bmi, smoking status, number of children, and region of residence). A medical insurance cost dataset is explored, cleaned, visualized, and preprocessed before being used to train an ElasticNet model (with hyper-parameter optimization). Pertinent performance metrics are evaluated and 6 major assumptions of Multiple Linear Regression are addressed. The model is saved and deployed through .pkl files.

**Language(s):** Python  
**Package(s):** Scikit-learn, Pandas, Seaborn, Statsmodels, Joblib  
**Software:** Jupyter Notebooks  

**Motivation:** This project was created to gain a deeper understanding of the entire ML model creation process starting from EDA and Data Cleaning to choosing the appropriate model, evaluating performance and major assumptions, and finally deploying it for others to use.

Viewing the Project
-------------------
The main file for this project is located in the root of this repository named _[Insurance Cost Multiple LinReg.ipynb](./Insurance%20Cost%20Multiple%20LinReg.ipynb)_. If you have **Jupyter Notebooks** installed, you can download the .ipynb file and view it there. If not, you can view the project using Google Colaboratory [here](https://colab.research.google.com/github/AvinashBisram/Machine-Learning/blob/master/Linear%20Regression/Insurance%20Cost%20Multiple%20LinReg.ipynb).  


About the Data
--------------
The data used in this project was a medical insurance cost dataset taken from Kaggle (to view the source see [here](https://www.kaggle.com/mirichoi0218/insurance)).  
The dataset was downloaded and saved as a CSV file, then imported into the project file using the Python **Pandas** module.  
The head of the dataframe can be seen below:  
![Head of Raw Dataset](./readMe%20images/raw_head.png)  
<br>
The raw dataset featured one dependent feature {charges} and six independent features {age, sex, bmi, children, smoker, region}.  


Exploratory Data Analysis
--------------------------
A basic EDA was performed on the dataset to gain a general idea of the format and data structure before moving on to Data Cleaning.  
Steps taken in this process included:
* Examining the head of the DataFrame
* Identifying the shape (1338 rows/records and 7 columns/features)
* Using .info() to look at non-null count and data type of each feature
* Calling .describe() to observe basic statistical information on the initial four numeric columns {age, bmi, children, charges}
* Checking the datatypes with .dtypes


Data Cleaning
-------------
The Data Cleaning process was done with the **Pandas** package.  
1. Features with the 'object' data type were investigated for formatting inconsistencies
    * Unique values for each feature: {sex, smoker, region} were identified. No formatting inconsistencies were found.
2. Duplicate records were identified and removed
    * 1 duplicate record was found and removed from the dataset
3. Missing data was identified
    * This dataset had no missing values so no further cleaning was done here


Visualization
--------------
Python's **Seaborn** package was used to visualize:
1. The distribution of values in the dataset for each feature
    * Histograms were created for the 3 quantitative features: {age, bmi, charges}
    * Count plots were created for the 4 qualitative features: {sex, children, smoker, region}
    * The type of data represented by each feature was discussed as well as noticeable trends and behavior of each plot
2. The relationship between each independent feature and the dependent feature ('charges')
    * Scatterplots were created for the 2 continuous independent features: {age, bmi}
    * Box plots were created for the 4 categorical independent features: {sex, children, smoker, region}
    * Intuition for using each visualization was discussed as well as significant observations in relation to the dependent feature.
A total of 13 visualizations were created in this process.


Data Transformation
--------------------
Nominal and Discrete qualitative features were identified and dummy encoding was used to convert them to numeric ones.  
The features expanded through dummy encoding included sex, smoker, and region.  
After converting all features to numeric data types, the correlation to the dependent feature was calculated for each and visualized with a heatmap.  
The smoker, age, and bmi features were predicted to be the most significant in explaining the behavior of charges (based on their relatively high correlations).


Preprocessing
-------------
The features of our cleaned dataset were separated into an X and y feature set.  
Python's **Scikit-learn** package was used to further divide these feature sets into a train-test split.  
A Standard Scaler was used to normalize the values in the X feature set.


Model Selection
---------------
The intuition behind using L1 or L2 Regularization in our model creation was discussed and a base **ElasticNet** model was created.  
GridSearchCV was used to find the optimal hyperparameters for our model (_alpha_ and _l1\_ratio_).
* An alpha value of 100 and l1_ratio of 1.0 (full Lasso Regularization) were found to be the optimal hyperparameters.

The new optimized model was fit on the training data and the calculated beta coefficients for each feature were compared and visualized using a bar plot.  
<br>
The absolute value coefficients can be seen below:  
![Visualization of Absolute Value Beta Coefficients](./readMe%20images/absolute_coefficients_viz.PNG)

Evaluating Performance
----------------------
The ElasticNet model was used to make predictions with the X_test records and overall performance was evaluated based on Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).  
MAE = 4351.718954689854  
RMSE = 6559.715600271675  
<br>
The meaning of these metrics in the context of the problem were discussed relative to the statistical description of all y_test values and real-life intuition.


Addressing Major Assumptions
-----------------------------
Each of the 6 major assumptions of Multiple Linear Regression were discussed:
1. Continuous Target Outcome --> True
2. Linearity between DV and IVs --> True
    * Trends identified during the visualization process were discussed here.
3. No Multicollinearity --> **False**
    * Correlation values were calculated and none were above the limit of 0.7.
    * Variance Inflation Factors (VIF) for each feature were calculated and those of {bmi, age} were deemed significant.
    * bmi = 11.359739, age = 7.696862
4. Normal Distribution of Residuals --> **False**
    * A Histogram was created to visualize the distribution of our model residuals. It appeared relatively normal with a slight elevation in the left tail.
    * A Q-Q Plot was created and significant deviation from the theoretical line of fit was observed at the tails and in the center.
5. Homoscedasticity --> True
    * To test for homogenous variance of residuals a scatterplot was created. The existence of clusters and lack of many points on the right side made it hard to distinguish but the variance appeared relatively homogenous (more research is needed on this subject).
6. Independence of Errors --> True
    * The Durbin-Watson statistic was calculated for our model residuals. (2.0009751588142572)

Summary of Major Assumption Tests: 4 TRUE, 2 FALSE  
Therefore, a Multiple Linear Regression model was not valid for the dataset in its current state.


Model Deployment
-----------------
Although our model didn't pass all major assumptions, Python's **Joblib** packagea was used to demonstrate how one could deploy and load the final model to make new predictions.
* The best estimator ElasticNet model was created and dumped with joblib, along with a StandardScaler (fit-transformed with all X records) and the column names (after dummy encoding) as .pkl files.
* New values for each independent feature were scaled and provided to a newly loaded model to make a prediction.
* Values given to model (before scaling): {age:35, bmi:18.5, 'children':0, 'sex_male':1, 'smoker_yes':0, 'region_northwest':0, 'region_southeast':0, 'region_southwest:0}. Predicted Insurance Cost = $3328.67.


Next Steps
-----------
If I wanted to expand on this project in the future I would...
* Create a better performing Regression model addressing the issues found in our major assumption tests
* Learn more about the benefits and drawbacks of different tests for each assumption
* Deploy the model as an API or web software