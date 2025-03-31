# CMPT-459-Project-Loan-Approval

## Data Preprocessing

### First 3 rows of the original data. 

|   Id |   Income |   Age |   Experience | Married/Single   | House_Ownership   | Car_Ownership   | Profession          | CITY      | STATE          |   CURRENT_JOB_YRS |   CURRENT_HOUSE_YRS |   Risk_Flag |
|-----:|---------:|------:|-------------:|:-----------------|:------------------|:----------------|:--------------------|:----------|:---------------|------------------:|--------------------:|------------:|
|    1 |  1303834 |    23 |            3 | single           | rented            | no              | Mechanical_engineer | Rewa      | Madhya_Pradesh |                 3 |                  13 |           0 |
|    2 |  7574516 |    40 |           10 | single           | rented            | no              | Software_Developer  | Parbhani  | Maharashtra    |                 9 |                  13 |           0 |
|    3 |  3991815 |    66 |            4 | married          | rented            | no              | Technical_writer    | Alappuzha | Kerala         |                 4 |                  10 |           0 |

The first thing done to the data was to remove any identifying attributes, so we removed the 'Id' attribute from the data. Next, we standardized all numerical features so that all numerical features had a Mean of 0 and a Standard Deviation of 1, using `StandardScaler()` from the `sklearn.preprocessing` library.

We used one-hot encoding and frequency-encoding the encode the categorical features into numerical features. `Married/Single`, `Car_Ownership` and `House_Ownership` were encoded using one-hot encoding because they had very limited unique attributes, For features like `CITY`, which had a large number of unique values, frequency encoding was preferred to keep the dataset manageable

```
def frequency_encode(original_dataframe, feature_to_encode, target):
    frequency_encoded = original_dataframe[feature_to_encode].value_count(normalize=True)
    original_dataframe[feature_to_encode] = original_dataframe[feature_to_encode].map(frequency_encoded)
    return original_dataframe
```

The code above is the function we used to perform feature encoding on `Profession`, `CITY` and `STATE` features.

### First 3 rows of the cleaned data (Post Preprocessing)

|    Income |       Age |   Experience |   Profession |      CITY |     STATE |   CURRENT_JOB_YRS |   CURRENT_HOUSE_YRS |   Risk_Flag |   Married/Single_single |   Car_Ownership_yes |   House_Ownership_owned |   House_Ownership_rented |
|----------:|----------:|-------------:|-------------:|----------:|----------:|------------------:|--------------------:|------------:|------------------------:|--------------------:|------------------------:|-------------------------:|
| -1.28314  | -1.5796   |   -1.18023   |     0.667449 | -0.173218 | -0.322059 |         -0.914131 |            0.716356 |           0 |                       1 |                   0 |                       0 |                        1 |
|  0.895457 | -0.583344 |   -0.0140667 |     0.226123 |  0.15846  |  1.01171  |          0.731036 |            0.716356 |           0 |                       1 |                   0 |                       0 |                        1 |
| -0.349269 |  0.940348 |   -1.01364   |     0.608247 | -0.8886   | -1.29173  |         -0.639936 |           -1.42798  |           0 |                       0 |                   0 |                       0 |                        1 |

### There were no NA's in our dataset

|       Attribute        |# NA |
|:-----------------------|----:|
| Income                 |   0 |
| Age                    |   0 |
| Experience             |   0 |
| Profession             |   0 |
| CITY                   |   0 |
| STATE                  |   0 |
| CURRENT_JOB_YRS        |   0 |
| CURRENT_HOUSE_YRS      |   0 |
| Risk_Flag              |   0 |
| Married/Single_single  |   0 |
| Car_Ownership_yes      |   0 |
| House_Ownership_owned  |   0 |
| House_Ownership_rented |   0 |

### Data Augmentation

The target variable `Risk_Flag` is unbalanced as you will see in `Exploratory Data Analysis`. SMOTE (Synthetic Minority Over-sampling Technique) will be used to handle this, though it is only used during the training phase for our classification models to not introduce any bias or data leakage to the test data and also the feature selection phase.

## Exploratory Data Analysis
### Dataset Summary

- **Rows**: 252,000
- **Columns**: 13
- **Target Variable**: `Risk_Flag` (1 = Risky, 0 = Not Risky)

### Missing Data

- There are **no missing values** in the dataset.

### Numerical Feature Distribution

| Feature                    | Mean        | Key Observations                                        |
|----------------------------|-------------|---------------------------------------------------------|
| **Income**                  | 4.99M       | High variance, indicating a wide range of incomes       |
| **Age**                     | 50          | Wide age range among applicants                         |
| **Experience**              | 10 years    | Average work experience                                 |
| **Current Job Duration**    | 6.3 years   | Average tenure in the current job                       |
| **Current House Duration**  | 12 years    | Average duration in current house                       |

### Categorical Feature Distribution

| Feature             | Categories       | Percentage Distribution             |
|---------------------|------------------|-------------------------------------|
| **Marital Status**   | Single, Married  | 89.79% Single, 10.21% Married       |
| **Car Ownership**    | Yes, No          | 30.16% Yes, 69.84% No              |
| **House Ownership**  | Owned, Rented    | 5.13% Owned, 92.02% Rented         |

### Target Variable Distribution

- **Risk 0 (Non-Risky)**: 87.7%
- **Risk 1 (Risky)**: 12.3%

This indicates a **significant class imbalance** in the dataset, with a higher proportion of non-risky applicants.

### Correlations

- **Experience vs. Current Job Duration**: Strong positive correlation of **0.65**, suggesting that applicants with more years of experience tend to stay longer in their current job.
- **No strong correlations** observed with the target variable (`Risk_Flag`).

### Key Insights

- Risky applicants tend to have:
  - Lower income
  - Less work experience

- **Class Imbalance**: The dataset is highly imbalanced, with **87.7%** of applicants classified as non-risky (`Risk_Flag = 0`) and only **12.3%** classified as risky (`Risk_Flag = 1`). Smote was used to address this.

## Feature Selection

We used `Mutual Information` to identify the most relevant features for predicting `Risk_Flag`. We used the method `mutual_info_classif` from the `sklearn` library.

### Mutual Information

| Feature                    | Mutual Info |
|----------------------------|-------------|
| Income                     | 0.159229    |
| Married/Single_single      | 0.035040    |
| House_Ownership_rented     | 0.031494    |
| Car_Ownership_yes          | 0.012715    |
| CITY                       | 0.008251    |
| CURRENT_HOUSE_YRS          | 0.007716    |
| Experience                 | 0.003699    |
| CURRENT_JOB_YRS            | 0.003656    |
| STATE                      | 0.003540    |
| Age                        | 0.002207    |
| Profession                 | 0.001516    |
| House_Ownership_owned      | 0.000776    |

### Plot of Mutual Information
![alt text](results/figures/mutual_info.png)

`Income` is by far the most important feature, with a MI of `0.159229`, almost 5x higher then the second highest feature. There is a noticeable drop off between `CURRENT_HOUSE_YEARS` and `Experience`, so we will cut off our features there, selecting the top 6 scoring features, `Income`, `Married/Single_single`, `House_Ownership_rented`, `Car_Ownership_yes`. `CITY`, `CURRENT_HOUSE_YEARS`.

In the next part, we train classification models using both a dataset only containing the features above and one with all original features, to compare if feature selection increases model efficiency and accuracy.


## Clustering

#### **K-Means Clustering**
- **How it was done**:
  - K-Means was applied with \(k\) ranging from 2 to 10.
  - The CH score was computed for each \(k\) to evaluate cluster quality.
- **Results**:
  - The CH score peaked at **2 clusters** (highest score: **72387.84**), indicating the best compactness and separation.
  - As \(k\) increased beyond 2, the CH score dropped steadily, suggesting that additional clusters led to less distinct groupings.

| Clusters | Calinski-Harabasz Score |
|----------|-------------------------|
| 2        | 72387.84                |
| 3        | 70377.95                |
| 4        | 70388.32                |
| 5        | 70092.02                |
| 6        | 69508.71                |
| 7        | 65652.09                |
| 8        | 63997.20                |
| 9        | 59803.75                |
| 10       | 56850.36                |

- **Plot**: The CH score as a function of \(k\) shows a clear peak at \(k = 2\):

![image](https://media.github.sfu.ca/user/2646/files/fda867ce-eb0b-43bd-a518-d427d74fb711)

- **Conclusion**: K-Means identified **2 clusters** as the optimal configuration for this dataset.

---

#### **DBSCAN Clustering**
- **How it was done**:
  - DBSCAN was applied with epsilon values ranging from 0.1 to 0.5.
  - The CH score was calculated for each epsilon to evaluate clustering quality.
- **Results**:
  - The CH score increased consistently as epsilon grew, reaching a score of **2066.73** at epsilon = 0.5.
  - As epsilon increased, the quality of clustering improved, but caution is needed to avoid potential overfitting for very high epsilon values.

| (epsilon)                             | CH Score |
|---------------------------------------|----------|
| 0.1                                   | 298.78   |
| 0.3                                   | 935.03   |
| 0.5                                   | 2066.73  |

- **Plot**: The CH score as a function of epsilon:

![image](https://media.github.sfu.ca/user/2646/files/0535a3b7-ac8c-4e17-8c1d-d167899de9f4)

- **Conclusion**: DBSCAN results indicate that clustering quality improves with higher epsilon, reaching a peak at epsilon = 0.5, but larger values of epsilon may lead to overfitting.

## Outlier Detection

### Methods Used
We applied **Isolation Forest** and **Local Outlier Factor (LOF)** to detect outliers across the datasets. The goal was to identify data points that deviate significantly from the norm.

### Key Findings

Refer to for IF analysis: https://github.sfu.ca/aca242/CMPT-459-Project-Loan-Approval/blob/main/results/metrics/isolation_forest_analysis.txt
Refer to for LOF analysis: https://github.sfu.ca/aca242/CMPT-459-Project-Loan-Approval/blob/main/results/metrics/local_outlier_factor_analysis.txt

#### Isolation Forest
- **Outliers Detected**: 10% of the data points.
- **Risk Distribution**: The risk ratio for outliers was **13.6%**, compared to **12.16%** for normal data points, with a difference of just **1.45%**.

![image](https://media.github.sfu.ca/user/2646/files/2b47753d-ed4f-4250-8d23-c6b397c07350)

#### Local Outlier Factor (LOF)
- **Outliers Detected**: 10% of the data points.
- **Risk Distribution**: The risk ratio for outliers was **19.91%**, compared to **11.45%** for normal data points, with a larger difference of **8.46%**.

![image](https://media.github.sfu.ca/user/2646/files/91451a10-c874-4cb1-9176-2e8af0d11034)

### Method Comparison

| **Method**                  | **Outliers Detected** | **Unique Outliers by Method** | **Common Outliers** |
|-----------------------------|-----------------------|-----------------------------|---------------------|
| **Isolation Forest**         | 10%                   | 22,626                      | 2,574               |
| **Local Outlier Factor**     | 10%                   | 22,626                      | 2,574               |

Both methods detected similar proportions of outliers. However, the **Isolation Forest** method flagged fewer unique points as outliers compared to **LOF**, yet both methods identified the same common set of **2,574 outliers**.

### Conclusion
Both methods identified around **10%** of the data as outliers. The risk distribution in LOF showed a higher deviation (8.46%) compared to Isolation Forest, though the overall conclusions were similar.

## Classification

We trained a total of eight models to predict `Risk_Flag` using two algorithms: `Random Forest` and `XGBoost`. Given that the `Risk_Flag` feature is imbalanced, we implemented `SMOTE`(Synthetic Minority Over-sampling Technique) during the training phase of some models.

The models were evaluated under the following conditions:

- `With SMOTE` vs. `Without SMOTE`
- `With Feature Selection` vs. `Without Feature Selection`

### Model Metrics

| Model                                  | SMOTE Used | Avg CV Score (Bal. Acc.) | Accuracy    | Balanced Accuracy | Precision | Recall  | F1-Score | AUC-ROC  | Time (s) |
|----------------------------------------|------------|--------------------------|-------------|-------------------|-----------|---------|----------|----------|----------|
| Random Forest (Feature Selection)      | False      | 0.7454                   | 0.8946      | 0.7449            | 0.5698    | 0.5471  | 0.5582   | 0.9312   | 181.24   |
| Random Forest (All Variables)          | False      | 0.7426                   | 0.8999      | 0.7421            | 0.6002    | 0.5336  | 0.5649   | 0.9383   | 187.29   |
| Random Forest (Feature Selection)      | True       | 0.8540                   | 0.8708      | 0.8561            | 0.4823    | 0.8367  | 0.6119   | 0.9351   | 357.33   |
| Random Forest (All Variables)          | True       | 0.8416                   | 0.8925      | 0.8426            | 0.5408    | 0.7767  | 0.6376   | 0.9389   | 410.10   |
| XGBoost (Feature Selection)            | False      | 0.5411                   | 0.8832      | 0.5473            | 0.6234    | 0.1033  | 0.1773   | 0.8569   | 8.16     |
| XGBoost (All Variables)                | False      | 0.6103                   | 0.8912      | 0.6178            | 0.6302    | 0.2564  | 0.3645   | 0.8904   | 10.77    |
| XGBoost (Feature Selection)            | True       | 0.8009                   | 0.8071      | 0.8067            | 0.3670    | 0.8061  | 0.5044   | 0.8632   | 16.95    |
| XGBoost (All Variables)                | True       | 0.8220                   | 0.8724      | 0.8267            | 0.4848    | 0.7663  | 0.5939   | 0.8716   | 25.09    |

### Scoring Context

In the context of our problem, which is predicting which customers present a loan risk to banks (`Risk_Flag`), the best model isn't simply which model has the highest `Accuracy`. The metrics most important to us in the table above are `Recall`, `Balanced Accuracy`, `AUC-ROC` and `F1-Score`. 
- `Recall` is the most important, it measures how well the model identifies risky customers (`Risk_Flag=1`) which also minimizes False Negatives (Failing to flag a risky customer).
    - False Negatives are the worst possible outcome in our scenario, as they would result in the bank issuing loans to risky customers.
    - We much rather have a False Positive (predicting a customer as risky when they’re not) than have a False Negative. Rather be safe then sorry.

### How did SMOTE affect the models?

Using SMOTE resulted in improvements in Balanced Accuracy, Recall, and F1-Score across all models that utilized it. Thus we can conclude that using SMOTE positively impacted the models ability to identify risky customers.

### How did Feature Selection affect the models? 

For Random Forest, feature selection had no significant impact on models without SMOTE, but it did improve all four key metrics when SMOTE was used. (Smote + Feature Seletion Model did better than Smote + No Feature Selection). 

As for XGBoost, Feature Selection improved Recall while making Precision worse in the SMOTE models. As for the non-SMOTE models we see that Feature Selection actually makes the model worse. 

All of the models which used Feature Selection were faster than those which did not.

### Best Model RF and XGBoost Models, best overall

The best RF model was the third one, using SMOTE and feature selection. It had the highest balanced accuracy, recall and f1-score while having almost identical AUC-ROC scores to the other RF models.

<p float="left">
  <img src="results/figures/roc_random_forest_fs.png" width="45%" />
  <img src="results/figures/conf_matrix_random_forest_fs.png" width="45%" />
</p>


The best XGBoost model is close between the 2 SMOTE models, but the Feature Selection model is slightly better, with an almost 4% higher Recall.

<p float="left">
  <img src="results/figures/roc_xgboost_fs.png" width="45%" />
  <img src="results/figures/conf_matrix_xgboost_fs.png" width="45%" />
</p>

The best performing model was the `Random Forest`, using `Feature Selection` and `SMOTE`. This model has the highest Recall. The model also has the highest `AUC-ROC` measure, which measures how the model can seperate between the positve and negative class. 

As seen in the Confusion Matrix above, the RF model has less False Negatives and False Positives, more True Positives and True Negatives, basically outperforming the XGBoost model everywhere. The only edge the XGBoost model has is that it is significantly faster (~17 Seconds vs ~360 Seconds). We will hypertune the random forest model.

## Hyper-parameter Tuning

Based off the results from last section, we will be tuning a RandomForest model on the feature selected data, using RandomizedSearchCV along with SMOTE during training and cross-validation to handle class imbalance. Similar to before, the data is split into training (80%) and testing (20%).

### Parameters
```
param_dist = {
    'rf__n_estimators': [50,100,150],
    'rf__max_depth': [None, 3,5,7,10],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4],
    'rf__max_features': ['sqrt', 'log2', None],
}
```

### Code for RandomizedSearchCV
```
pipeline = Pipeline([('smote', SMOTE(random_state=1)), ('rf', RandomForestClassifier(random_state=1))])
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, 
                                   n_iter=10, cv=cv, verbose=2, random_state=1, n_jobs=-1,scoring='recall')
```

The above code ensures that SMOTE is used in each fold so there is no data leakage on the validation data. We are tuning the model to find the highest `recall` as possible.

The RandomizedSearch takes 884.84 seconds to run, and finds the following parameters as the best:

| Parameter              | Value   |
|------------------------|---------|
| rf__n_estimators       | 150     |
| rf__min_samples_split  | 10      |
| rf__min_samples_leaf   | 1       |
| rf__max_features       | None    |
| rf__max_depth          | None    |



### Tuned Random Forest Metrics

| Metric             | Value              |
|--------------------|--------------------|
| Accuracy           | 0.8637             |
| Balanced Accuracy  | 0.8615             |
| Precision          | 0.4675             |
| Recall             | 0.8585             |
| F1-Score           | 0.6053             |
| ROC-AUC            | 0.9365             |
| Time (s)           | 884.84             |

### Tuned Random Forest Plots
<p float="left">
  <img src="results/figures/roc_tuned.png" width="45%" />
  <img src="results/figures/conf_matrix_tuned.png" width="45%" />
</p>

From the above, we can see that the tuned model is the best one found so far, with less False Negatives, More True Positives and similar True Negatives and True Negatives. Compared to the best model in the previous step, the tuned model has a higher Recall, F1-Score, Balanced Accuracy and slightly higher ROC-AUC. The only downfall to the tuned model is that it takes ~885 seconds to tune/train compared to ~360 seconds to train the Random Forest with default parameters.

We are happy with the results of this tuned model. We are correctly identifying 38,263 customers as No Risk, 5,268 as High Risk while only incorrectly flagging 6,000 No Risk customers as high risk and most importantly only classifying 868 risky customers as no risk. 

## Challenges & Limitations 

The biggest limitation is the imbalanced data. While we were able to work around this by using SMOTE during model training, it's not always a perfect solution. Generating synthetic samples can sometimes lead to overfitting, causing the model to learn noise or overly specific patterns that do not generalize well. Also, we want to test the model on real data, not synthetic data, so we weren't able to test the model on the minority class as much as we'd like. Another limitation is the encoding methods. Encoding the categorical variables isn't always perfect and frequency encoding isn't a perfect solution. Final limitation was dataset size, the data is massive (>200k rows), we probably could have taken a sample to get more efficient models.

## Conclusion

### Insights 

1. **Income Correlates with Risk**: Higher income applicants tend to be classified as lower risk. This suggests that income is a strong indicator of financial stability, directly impacting the likelihood of loan repayment.

2. **Housing and Vehicle Ownership**: Applicants who own a house or car are less likely to be high-risk, highlighting that asset ownership signals financial stability. Those without these assets are more likely to be considered high-risk.

3. **Marital Status and Loan Approval**: Marital status does not significantly impact loan approval risk. Single and married applicants show similar trends in loan risk classification, suggesting that financial factors are more important than relationship status.

4. **Work Experience and Stability**: Applicants with longer work experience are less likely to be high-risk. This suggests that lenders may prioritize stable employment as a predictor of consistent income and repayment ability.

5. **Class Imbalance**: The dataset shows a high proportion of low-risk applicants, which indicates that the loan approval system may need better methods to detect and evaluate high-risk applicants in a skewed dataset.


### Lessons Learned

We learned that working with large datasets in data mining came with benefits and challenges. On one hand, more data often improves model performance, but also introduced issues like much longer processing times, and some pretty ugly plots since we had so many points, we definely learned about the importance of sampling data, because looking back it would have helped quite a bit. Hyperparameter tuning, especially on large datasets, can take significant time as well as just training a Random Forest Model with default parameters. I couldn't believe how much faster the XGBoost models were compared to the Random Forest Models. 

We have also learned that the problem context plays a crucial role; for example, in our case, an imbalanced dataset (Where Risk_Flag = 1 was the most important, and the minority class) made us prioritize recall over total accuracy to better handle the underrepresented class. Also, addressing the imbalanced data, we learned of techniques like SMOTE.

Overall, we learned not to under-estimate how long it could take to complete a data-mining project from start to finish, including even the time it takes to run some of these models. We also learned of some techniques not taught in class such as SMOTE and Frequency Encoding due to our unique dataset. 

## Visualizations

### EDA

![image](https://media.github.sfu.ca/user/2646/files/238e985e-95c4-4786-9d64-eff88f3e10fa)
![image](https://media.github.sfu.ca/user/2646/files/3d70d5a2-78aa-48b6-97e1-fd9c424927df)
![image](https://media.github.sfu.ca/user/2646/files/5f271e5a-52ef-485c-89ba-43b7a93b1285)
![image](https://media.github.sfu.ca/user/2646/files/0fd96f0d-bab4-4860-b754-2b389d975c59)
![image](https://media.github.sfu.ca/user/2646/files/89829051-b46a-4d29-8a6d-338c83d3aafc)
![image](https://media.github.sfu.ca/user/2646/files/a3b4acf3-9f02-4606-8721-6fcf20d7fe62)
![image](https://media.github.sfu.ca/user/2646/files/228b837d-fccc-4b16-9c86-b82337dd1a17)
![image](https://media.github.sfu.ca/user/2646/files/895093c7-7848-4b9f-bad0-a6756d2c27e1)
![image](https://media.github.sfu.ca/user/2646/files/c9db78b4-9977-4df1-898c-f5f2623233de)
![image](https://media.github.sfu.ca/user/2646/files/3d70d5a2-78aa-48b6-97e1-fd9c424927df)

Refer to for EDA summary: https://github.sfu.ca/aca242/CMPT-459-Project-Loan-Approval/blob/main/results/metrics/eda_summary.txt

### Clustering

![image](https://media.github.sfu.ca/user/2646/files/a3f2638d-aba0-4f5e-ad71-62f34ab643b2)
![image](https://media.github.sfu.ca/user/2646/files/f150376d-e0a4-4fe6-adbc-a2d2440d6d17)
![image](https://media.github.sfu.ca/user/2646/files/7766387d-ef3c-4ba7-be53-1d7a87c635c4)
![image](https://media.github.sfu.ca/user/2646/files/5c163784-7fab-44d0-b77c-3e90e03f9b47)
![image](https://media.github.sfu.ca/user/2646/files/c9db78b4-9977-4df1-898c-f5f2623233de)

### Outlier Detection

![image](https://media.github.sfu.ca/user/2646/files/583dfd93-92b6-4748-a78c-fefc6ef9eb18)
![image](https://media.github.sfu.ca/user/2646/files/162c5754-8857-40fb-bcc2-b6b240bf1433)

Refer to for IF analysis: https://github.sfu.ca/aca242/CMPT-459-Project-Loan-Approval/blob/main/results/metrics/isolation_forest_analysis.txt

Refer to for LOF analysis: https://github.sfu.ca/aca242/CMPT-459-Project-Loan-Approval/blob/main/results/metrics/local_outlier_factor_analysis.txt

### Feature Selection

![image](https://media.github.sfu.ca/user/2646/files/82d83dbd-fef4-4bf5-9fa5-5ec655d87871)

### Classification

![image](https://media.github.sfu.ca/user/2646/files/95fab57f-af47-4f73-8afc-91e3e3129076)
![image](https://media.github.sfu.ca/user/2646/files/5f58b797-dfcb-4525-85c3-8381e6f44e03)
![image](https://media.github.sfu.ca/user/2646/files/957d608e-f8cf-4840-b5ba-4ae4c98759fd)
![image](https://media.github.sfu.ca/user/2646/files/7084bc29-06cf-402c-8a39-11e9c17edb5f)
![image](https://media.github.sfu.ca/user/2646/files/ec9e54c0-fc44-4ba7-b346-adbd04c42251)
![image](https://media.github.sfu.ca/user/2646/files/757e2926-a14c-4d7f-8a66-98357aaef5fc)
![image](https://media.github.sfu.ca/user/2646/files/44226d09-a0bd-458a-b868-4959b83ac11b)
![image](https://media.github.sfu.ca/user/2646/files/5789500d-afaa-4886-8461-9330b15cc8f9)
![image](https://media.github.sfu.ca/user/2646/files/947a817e-4da9-487a-b0f9-108124d0db64)
![image](https://media.github.sfu.ca/user/2646/files/3cafda15-1e6a-431f-9583-b254005fb4a9)
