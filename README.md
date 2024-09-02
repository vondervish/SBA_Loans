# SBA_Loans


# SUMMARY OF WORK

The above project is a Classifier problem where we have to predict whether the loan will be defaulted or not defaulted. We are given a data set of 799356 records and 18 columns (excluding the 'index' and 'MIS_Status'). MIS_Status is the target column. We have 18 predictor columns -'City', 'State', 'Zip', 'Bank', 'BankState', 'NAICS', 'NoEmp', 'NewExist', 'CreateJob', 'RetainedJob', 'RevLineCr', 'LowDoc', 'DisbursementGross', 'BalanceGross', 'GrAppv', 'SBA_Appv',' 'UrbanRural','FranchiseCode'. We check for the missing values in the columns where it is found that 'City' has 25 missing values, 'State' has 12 missing values, 'Bank' has 1402 missing values, 'BankState' has 1408 missing values, NewExist has 114 missing values, RevLineCr has 4025 missing values and LowDoc has 2300 missing values.

## Splitting the data
We split the data into X_train, X_valid, X_test in the ratio of 60:20:20. We use the training dataset to training the model and evaluate the model using the validation dataset. We use test dataset (unseen data) to display the performance metrics of the best model.

## Data Cleaning
In data cleaning we have initially, created a function to clean the data and use it further during scoring of new data. We have replaced missing values in columns with 0 and 'Missing' if the column data type is numeric or object respectively. Further, in RevLineCr - Revolving line of credit(either Yes or No) is a categorical column. We have replaced values of 0 or N with N and 1 or Y with Y and any value other than this with 'Missing'.Similarly, We replace values of 0 or N with N and 1 or Y with Y and any value other than this with 'Missing'in the column of LowDoc.In the column FranchiseCode, we replace the values other than 0 or 1 with 1.
We create new column, sector by extracting the first two digits from the NAICS. We create a dictionary of the digits and corresponding sector and map them accordingly to create the column sector.
We then extract the first 4 digits from the Zip column as this captures a lot more information by reducing the Noise and allowing to focus on the underlying pattern.
We then drop the column 'BalanceGross'. We can see that the values of BalanceGross other than 0 are very less (75th percentile of 'BalanceGross' is also 0), and hence is not contributing in model prediction. We can also see that the number of records of 'BalanceGross' > 0 in 799356 records is only 13, which is highly negligible which may result in the Noise in data. Hence, this fails to capture any important information and removing this column can increase the model performance.
We perform log transformations on the amount columns - 'DisbursementGross', 'SBA_Appv', 'GrAppv', to remove the skewness if there is any. We the call the function data_cleaning on X_train, X_valid, X_test to perform data cleaning on the three data sets separately.

## Model performance before feature Engineering and hyper tuning using default parameters
We have created a gbm model after performing data cleaning and calculated it's AUC which is 0.805225. Our target is to introduce new features (atleast 10) to improve the model performance.

## New feature Addition
We create a function 'new_features_1' to add engineering features. In function 'new_features_1' we added 5 new features.
'pct_job_retained' - percentage of job retained which is calculated by taking the percentage of 'RetainedJob' with respect to 'CreatedJob'.
'SBA_pct_app' - percentage of SBA_loan_approved which is calculated by taking the percentage of 'SBA_Appv' as a percentage of 'GrAppv'.
'disb_greater_app' - gives a value 1 if the DisbursementGross is greater than GrAppv and 0 otherwise.
'NewExist_Franchisecode_Comb'- It is a combination column of 'NewExist' and 'FranchiseCode'.
'Bank_BankState_Comb'- It is a combination column of 'Bank' and 'BankState'.
We call this function new_features_1 on the datasets X_train, X_valid, X_test, to add the new features.
We create another function 'new_features_2' to add 4 new engineering features.
'Avg_SBAappv_bank'- We create this column by calculating grouped mean of the SBA_Appv on the Bank.
'Avg_GrAppv_sector' - We create this column by calculating grouped mean of the GrAppv on sectors.
'Avg_GrAppv_State' - We create this column by calculating grouped mean of the GrAppv on State.
'Avg_Disb_Bank'- We create this column by calculating grouped mean of DisburseGross on Bank.
We use this function add this new features in X_train.
We then create dictionaries 'Bank_SBAappv_mapping','sector_GrAppv_mapping','State_GrAppv_mapping','Disb_Bank_mapping' to store the values created during the new feature addition in the X_train. We map these stored values in the X_valid and X_test separately, by mapping them based on Bank, sector, State accordingly.
We create another function add_interaction_terms where we create interaction terms of - ('UrbanRural' and 'FranchiseCode') and ('disb_greater_app' and 'FranchiseCode'). We add these interaction terms in X_train, X_valid, X_test by calling the function 'add_interaction_terms'.
By this we completed our features addion. In total we have added 11 engineering features into the data sets.

## Changing the dataframe
We now change the datasets to H2O dataframe and save them as X_train_h2o, X_valid_h2o, X_test_h2o.

## Changing the datatype
We then change the datatype of the columns as per the requirement. We create a function 'changing_datatype'.
'City', 'State', 'Zip', 'Bank', 'BankState', 'NAICS', 'NewExist', 'sector', 'NewExist_Franchisecode_Comb', 'Bank_BankState_Comb','UrbanRural', 'UrbanRural FranchiseCode', 'disb_greater_app', 'FranchiseCode', 'disb_greater_app FranchiseCode' - We convert the dataype of these columns to categorical type using asfactor(). We change the rest of the columns dataype to asnumeric(). We call this function X_train_h2o, X_valid_h2o, X_test_h2o.
We then change the datatype of 'MIS_Status' to categorical and drop the 'index' column as it will not be used in the model training.

## Model performance after feature Engineering and hyper tuning using default parameters
We can see than on addition of new features:'pct_job_retained', 'SBA_pct_app', 'NewExist_Franchisecode_Comb', 'Bank_BankState_Comb', 'Avg_SBAappv_bank', 'Avg_GrAppv_sector', 'Avg_GrAppv_State', 'Avg_Disb_Bank', 'UrbanRural', 'UrbanRural FranchiseCode', 'disb_greater_app', 'FranchiseCode', 'disb_greater_app FranchiseCode', the model performance(using AUC metric) increased from 0.805225 to 0.819027

## Hyperparameter tuning and Best model parameters
We perform grid search on the X_train_h2o using 54 combinations as mentioned above. We train the model on X_train_h2o and perform evaluation on X_valid_h2o. We use 'auc' performance metric to evaluate the model and find the optimal parameters. We save the best model as best_model_h2o.

## SUMMARY OF MODEL PERFORMANCE
We can see that the optimal params are:
      {'max_depth': 15,
      'learn_rate':0.01,
      'min_rows': 15,
      'col_sample_rate': 0.75,
      'sample_rate': 0.75}
The 'auc' on X_valid_h2o is 0.8426.
The performance metrics on X_test_h2o which is an unseen data to the model:
Best F1 threshold: 0.24488
F1: 0.55052
Model AUC: 0.84223
The probability threshold where we get the maximum F1 score in 0.24488 and corresponding max F1 = 0.55052
The 'auc' for this model on X_test_h2o is 0.84223.
The confusion matrix on test data is :
Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.24488
           0               1               Error Rate
0     115320.0     16579.0      0.1257 (16579.0/131899.0)
1     11052.0     16921.0      0.3951 (11052.0/27973.0)
Total   126372.0     33500.0     0.1728 (27631.0/159872.0)

## SUMMARY OF MODEL FINDINGS
From the confusion matrix, we can see that the model could predict class_1 (default) correctly in 16921 instances out of 27973 instances and rest 11052 instances couldn't be predicted accurately.

## Model Interpretability
The contribution of features in the model can be known by model interpretability.
We plot the Permutation Feature importance of the best_model_h2o on test data. We can see that BankState, SBA_pct_app, NewExist_Franchisecode_Comb, State, Avg_SBAappv_bank are the top 5 important features. Using permuation feature importance plot we can say the features which are important in the model but this doesnot give us any idea about the direction of probability prediction.This plot shows whether the feature is important enough to change the prediction value of the given target variable, but it doesn't provide any information regarding the direction. This is because the permutation feature importance involves a process where single feature's values are shuffled and the effect of it is measured by degradation in the model performance. This breaks the relationship between the feature and the target variable and hence we can only see the magnitude change in model's performance and not whether the feature was actually associated with high or low predicted probabilities.

To know the direction of probability we can do Shap plots. We have plotted Shap Summary. The top most important feature is 'SBA_pct_app'. The red points which denote high values of SBA_pct_app, mostly have negative shap values and those with blue points which denote low values of SBA_pct_app have positive shap values. This means that high values of SBA_pct_app will result in decrease in 'p1' probability, thus contributing more to prediction value of 0 (not default). Similarly, NewExist_Franchise_Comb have a cluster of values with negative shap values and also a range of shap values with positive shap. 'UrbanRural FranchiseCode', which is an interaction term, it can be seen that the blue points, with low normalised feature values, have negative shap contributions, increasing the chance of prediction value of 0 (Not_defaulted). A few points with high normalised values and medium normalised values, have positive shap contributions, thus increasing the chance of prediction value of 1 (Defaulted).

## Individual observations analysis using Shapley values.
We have plotted two records for each of the scenarios with significant probability:
Label 0 is correctly identified
Label 0 is identified as 1
Label 1 is correctly identified
Label 1 is identified as 0
The plots and the corresponding explanations are mentioned above.
Through these plots we are able to see how features in individual records are contributing the prediction class. We have used instances with significant probability. For example, for a record where actual 'MIS_Status' = 0 and predicted 'MIS_Status' = 0, i.e. not defaulted, or the case where actual 'MIS_Status' = 1 and predicted 'MIS_Status' = 1 (Defaulted), we have plotted a case with residual error of nearly 0, i.e. 'p1' value much below the threshold probability or much above the threshold probability respectively. In these cases, we can strongly say that the model has made significantly correct prediction. Similarly, cases with highly positive residuals (actual 'MIS_Status' = 1 and predicted 'MIS_Status' = 0) or highly negative residuals(actual 'MIS_Status' = 0 and predicted 'MIS_Status' = 1), we can strongly say that the model has made significantly incorrect predictions.

## Residual Analysis
We Calculated the residuals for each record. We then plotted the shap summary plots for top 1000 positive residuals and top 1000 negative residuals.
Plot with positive residuals- These are the top observations that model predicted very low probability of default while actual label indicated default. In this plot, we can see that the shap values of the features are mostly in the negative range as compared to positive shap values, which brings down the predicted probablity.
Plot with negative residuals - These are the top observations that model predicted very high probability of default while actual label indicated not default. In this plot, we can see that the shap values of the features are mostly in the positive range as compared to negative shap values, which pushes up the predicted probablity.

## SUMMARY OF RECOMMENDATIONS
One of the remedies to bring down residuals and increase the model performance is by increasing the data. In this particular case, we are nearly dealing with 800000 records, but after the split to train, valid, test data, we are left only with 60% of the data for training. As we can see, the cardinality of certain columns like City, State, Zip is very high. Hence, there can be a situation where there are no enough number of cases for each of the unique values to identify a certain pattern in the data.
Also, providing more information, i.e. more features can also aid the performance of the model. These features can be either directly used or can be used after feature engineering. Adding important or relevant features can add more information and hence improve the model performance. We can use Surrogate model to improve the interpretability of the model. Using Surrogate model, we can explain the most important splits in the model and hence improve the expalainability.
Using Stacking Ensemble learning technique - We can use different base models like logistic, RandomForest for prediction and use either blending or a meta model for final prediction. This will deal any possibility of bias and variance being created and can uplift the model performance.
Knowing the Cost of classifying/predicting 'Default' case to 'Not_Default' is crucial. If the cost is very high we can try to increase the threshold value so that we are able to capture most of the 'Default' cases possible. However, this cost is to be more than the computational cost.
Finally, better understanding of the data by focussing and understanding Exploratory data analysis can improve the approach in dealing with features and feature engineering.
