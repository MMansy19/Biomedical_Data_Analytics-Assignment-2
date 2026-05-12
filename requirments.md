Overview
Choose any medical dataset of your choice in CSV format from a trusted source (e.g., Kaggle,
PhysioNet, UCI Repository). Your dataset must contain at least 5 numeric columns, 300+ rows of
real biomedical or clinical data, and a clearly defined target variable suitable for classification.
Apply what you learned in Tutorials to preprocess the data, train machine learning models, evaluate
their performance using the correct metrics, and interpret the results. Your notebook should tell a
clear, well-explained machine learning story.
Note: Start your notebook by stating the dataset name, its source URL, number of rows and columns,
the target variable (what you are predicting), and why you chose this dataset.
Part 1 — Data Preparation
Before building any model you must prepare your data. Perform the following steps and briefly
explain each decision:
• Handle any missing or impossible values using an appropriate method. Justify your choice.
• Identify the target column (the label you want to predict) and separate it from the features.
• Split your dataset into a training set (80%) and a test set (20%) using train_test_split. Set a
random_state for reproducibility.
• Apply feature scaling to all numeric features. State which method you used (Min-Max or
Standard Scaler) and explain why it is important before training the models in this part.
Reminder: Fit the scaler on the training set only, then transform both train and test sets. Never fit on
the test set.
Part 2 — Model Training
Train the following three classification models on your training data. For each model, write a short
paragraph (3–5 sentences) explaining how the algorithm works before running any code.
Model 1: Logistic Regression
• Explain what the logistic (sigmoid) function does and why it is used for classification instead
of linear regression.
• Train a Logistic Regression model using scikit-learn with default parameters.
• Print the model’s coefficients and interpret which two features have the strongest positive and
negative influence on the prediction.
Model 2: Support Vector Machine (SVM)
• Explain the concept of the hyperplane, margin, and support vectors in your own words.
• Train an SVM classifier using a linear kernel.
• Try at least two values of the C parameter (e.g., C=0.01 and C=1). Report training accuracy
for each and explain which you choose and why, referring to the trade-off between margin
width and classification errors on noisy medical data.

Model 3: Your Choice
• Choose one additional supervised classification algorithm not listed above (e.g., K-Nearest
Neighbors, Decision Tree, or Naive Bayes).
• Briefly explain how it works.
• Train it on the same training set with default parameters.
Part 3 — Model Evaluation
Evaluate all three trained models on the test set using the metrics from Tutorial 7. Do NOT evaluate
on the training set.
• Compute and display the Confusion Matrix for each model as a heatmap. Label the axes
clearly.
• For each model compute and report: Accuracy, Precision, Recall (Sensitivity), F1-Score, and
Specificity. Show these in a single comparison table.
• Plot the ROC Curve for each model on the same graph and report the AUC score for each.
• Identify the best-performing model. Justify your answer using at least two metrics, not just
accuracy.
Clinical Note: Remember from the tutorial that in medicine, Recall (Sensitivity) is often more
important than Accuracy. Missing a disease is worse than a false alarm. Discuss whether this applies
to your dataset.
Part 4 — Cross-Validation
Apply k-fold cross-validation to your best model from Part 3.
• Use 5-fold cross-validation (k=5) as shown in the tutorial.
• Report the accuracy score for each fold and the mean and standard deviation across all folds.
• Compare the cross-validation mean accuracy with the single test set accuracy from Part 3.
Are they similar or different? What does this tell you about how reliable your model is?
• Explain in 2–3 sentences why cross-validation gives a more reliable performance estimate
than a single train/test split, especially for medical datasets where data is often limited.