# Credit Risk Scoring

This project is inspired in a ML Zoomcamp.

In this project I created a model to predicts whether a bank should lend loan to a client or not. The bank takes these decisions based on the historical record. The dataset used has [credit scoring data](https://github.com/gastonstat/CreditScoring/blob/master/CreditScoring.csv). 

In the credit scoring classification problem,

* if the model returns 0, this means, the client is very likely to payback the loan and the bank will approve the loan.
* if the model returns 1, then the client is considered as a defaulter and the bank may not approval the loan.

I have followed the steps described:

* 👀 Prepare data

    **Main Conclusions** : This step included data obtention and some procedures of data preparation, namely look at the data, make columns names look uniform, handle missing values and removing rows which the objective variable is unkown.
    * Load the data 
    * Reformat categorical columns (status, home, marital, records, and job) by mapping with appropriate values.
    * Replace the maximum value of income, assests, and debt columns with NaNs.
    * Replace the NaNs in the dataframe with 0
    * Prepare target variable status by converting it from categorical to binary, where 0 represents ok and 1 represents default.


* 🐱‍👤 Setting up the validation framework (split between train, validation and test)

    **Main Conclusions** : I have splitted the dataset using Scikit-Learn into train, validation and test.
    * Split the data in a two-step process which finally leads to the distribution of 60% train, 20% validation, and 20% test sets with random seed to 11


 * 0️⃣1️⃣ One-hot Encoding

    **Main Conclusions** : I have used Scikit Learn - Dict Vectorizer - to encode categorical features. This method represents each category of a variable as one column, and a 1 is assigned if the value belongs to the category or 0 otherwise.


* 👩‍💻 Use Decision Trees to identify to which clients the bank should lend loan

    **Main Conclusions** : The decision trees make predictions based on the bunch of if/else statements by splitting a node into two or more sub-nodes. With versatility, the decision tree is also prone to overfitting. One of the reason why this algorithm often overfits because of its depth. It tends to memorize all the patterns in the train data but struggle to performs well on the unseen data (validation or test set).

    To overcome with overfitting problem, I have reduced the complexity of the algorithm by reducing the depth size.
    The decision tree with only a single depth is called decision stump and it only has one split from the root.
 
    I have made some parameter tuning, namely in two features, max_depth and min_samples_leaf that have a greater importance than other parameters. First I tunee max_depth parameter and then moved to tuning other parameters. Finally, a dataframe is created with all possible combinations of max_depth, min_sample_leaf and the auc score corresponding to them. These results are visualized using a heatmap by pivoting the dataframe to easily determine the best possible max_depth and min_samples_leaf combination. Finally, the DT is retrained using the identified parameter combination.

* 👩‍💻 Use Random Forest to identify to which clients the bank should lend loan
 
  **Main Conclusions**: Random Forest is an example of ensemble learning where each model is a decision tree and their predictions are aggregated to identify the most popular result. Random forest only select a random subset of features from the original data to make predictions. In random forest the decision trees are trained independent to each other.

  I have also made parameter tuning  regarging min_sample_leaf parameter and  number of estimators.

* 👩‍💻 Use XGBoost to identify to which clients the bank should lend loan
    
    **Main Conclusions**: Unlike Random Forest where each decision tree trains independently, in the Gradient Boosting Trees, the models are combined sequentially where each model takes the prediction errors made my the previous model and then tries to improve the prediction. This process continues to n number of iterations and in the end all the predictions get combined to make final prediction.To train and evaluate the model, I needed to wrap my train and validation data into a special data structure from XGBoost which is called DMatrix. 
    This data structure is optimized to train xgboost models faster.XGBoost has various tunable parameters but the three most important ones and the ones that I have tuned are:

    * eta (default=0.3) - It is also called learning_rate and is used to prevent overfitting by regularizing the weights of new features in each boosting step. range: [0, 1]
    * max_depth (default=6) - Maximum depth of a tree. Increasing this value will make the model mroe complex and more likely to overfit. range: [0, inf]
    * min_child_weight (default=1) - Minimum number of samples in leaf node. range: [0, inf]

    For XGBoost models, there are other ways of finding the best parameters as well but the one implemented by me was the following:

    * First find the best value for eta
    * Second, find the best value for max_depth
    * Third, find the best value for min_child_weight

    Other useful parameter are:

    * subsample (default=1) - Subsample ratio of the training instances. Setting it to 0.5 means that model would randomly sample half of the trianing data prior to growing trees. range: [0, 1]
    * colsample_bytree (default=1) - This is similar to random forest, where each tree is made with the subset of randomly choosen features.
    * lambda (default=1) - Also called reg_lambda. L2 regularization term on weights. Increasing this value will make model more conservative.
    * alpha (default=0) - Also called reg_alpha. L1 regularization term on weights. Increasing this value will make model more conservative.

* 🎆 Using the model

    **Main Conclusions**: After finding the best model, it was trained with training and validation partitions (x_full_train) and the final accuracy was calculated on the test partition. If there is not much difference between model auc scores on the train as well as test data then the model has generalized the patterns well enough.
    
    XGBoost models perform better on tabular data than other machine learning models but the downside is that these model are easy to overfit cause of the high number of hyperparameter. Therefore, XGBoost models require a lot more attention for parameters tuning to optimize them.