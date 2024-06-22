# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:34:02 2024

@author: user
"""

import pandas as pd
from math import sin, cos, pi
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from xgboost import XGBClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import joblib
import numpy as np
#%%

data = pd.read_csv('eurusd_0.95.csv', index_col=0)
# Convert 'Entry Index' to datetime and extract the hour
data['Entry Index'] = pd.to_datetime(data['Entry Index'])
data['hour'] = data['Entry Index'].dt.hour

# Apply sine and cosine transformation to the hour
data['hour_sin'] = data['hour'].apply(lambda x: sin(2 * pi * x / 23))
data['hour_cos'] = data['hour'].apply(lambda x: cos(2 * pi * x / 23))

# Creating new features
data['hour_sin_atr'] = data['hour_sin'] * data['atr']
data['hour_cos_atr'] = data['hour_cos'] * data['atr']
data['tick_volume_atr_ratio'] = data['tick_volume'] / data['atr']
data['atr_tick_volume'] = data['atr'] * data['tick_volume']
data['tick_volume_per_atr'] = data['tick_volume'] / data['atr']
data['log_tick_volume'] = np.log1p(data['tick_volume'])  # log(1 + tick_volume) to avoid log(0)
data['atr_squared'] = data['atr'] ** 2


# Select relevant features and the label
features = ['hour_sin', 'hour_cos', 'tick_volume', 'atr', 'hour_sin_atr', 'hour_cos_atr', 'atr_tick_volume', 'tick_volume_per_atr', 'log_tick_volume', 'atr_squared', 'tick_volume_atr_ratio']
label = 'PnL'

# Splitting the data into training and testing sets while keeping the sequence in mind
# Let's use the last 20% of the data as the test set
date_cutoff = '2023-4-3'
split_ratio = 0.6
split_index = int(len(data.loc[:date_cutoff]) * split_ratio)




train_data = data.loc[:split_index]
test_data = data.iloc[split_index:]
test_data = test_data.loc[:date_cutoff]
out_of_sample_data = data.loc[date_cutoff:]
# Split into features (X) and label (y)
X_train = train_data[features]
y_train = train_data[label]
X_test = test_data[features]
y_test = test_data[label]
X_out = out_of_sample_data[features]
y_out = out_of_sample_data[label]

X_train.head(), y_train.head()

#%% Random Forest


# Initialize the RandomForest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
rf_confusion_matrix = confusion_matrix(y_test, y_pred)
accuracy, classification_rep

# Calculate the probabilities of the positive class
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, rf_prob)
roc_auc = auc(fpr, tpr)

# Save the trained model
joblib.dump(rf_model, 'random_forest_model_new.joblib')

# Plot ROC curve
plt.figure(figsize=(16,9))
plt.subplot(1,2,1)

plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Random Forest')
plt.legend(loc="lower right")
# plt.show()

# Extracting one tree from the Random Forest model
# single_tree = rf_model.estimators_[0]

# # Plotting the tree
# plt.figure(figsize=(20,10))
# plot_tree(single_tree, filled=True, feature_names=features, max_depth=3)
# plt.title("Decision Tree from Random Forest")
# plt.show()
precision, recall, _ = precision_recall_curve(y_test, rf_prob)
pr_auc = average_precision_score(y_test, rf_prob)

# Plot the Precision-Recall curve
plt.subplot(1,2,2)
plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (AUC = %0.2f)' % pr_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.show()

# Predict on the out of sample set
y_pred_out = rf_model.predict(X_out)
rf_prob_out = rf_model.predict_proba(X_out)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_out, rf_prob_out)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(16,9))
plt.subplot(1,2,1)

plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Random Forest OUT')
plt.legend(loc="lower right")
# plt.show()

# Extracting one tree from the Random Forest model
# single_tree = rf_model.estimators_[0]

# # Plotting the tree
# plt.figure(figsize=(20,10))
# plot_tree(single_tree, filled=True, feature_names=features, max_depth=3)
# plt.title("Decision Tree from Random Forest")
# plt.show()
precision, recall, _ = precision_recall_curve(y_out, rf_prob_out)
pr_auc = average_precision_score(y_out, rf_prob_out)

# Plot the Precision-Recall curve
plt.subplot(1,2,2)
plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (AUC = %0.2f)' % pr_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve OUT')
plt.legend(loc="lower right")
plt.show()

#%%XGBoost

# Initialize models
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
# Train and evaluate XGBoost
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_class_rep = classification_report(y_test, xgb_pred)
xgb_confusion_matrix = confusion_matrix(y_test, xgb_pred)


xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, xgb_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(16,9))
plt.subplot(1,2,1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC XGBoost')
plt.legend(loc="lower right")
# plt.show()

# Plotting the first tree with the default style
# xgb.plot_tree(xgb_model, num_trees=0)
# plt.rcParams['figure.figsize'] = [50, 10]
# plt.savefig("xgb_tree_high_res.png", dpi=300)  # Save as high-resolution PNG
# plt.show()
precision, recall, _ = precision_recall_curve(y_test, xgb_prob)
pr_auc = average_precision_score(y_test, xgb_prob)
plt.subplot(1,2,2)
plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (AUC = %0.2f)' % pr_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.show()

# Predict on the out of sample set
y_pred_out = xgb_model.predict(X_out)
xgb_prob_out = xgb_model.predict_proba(X_out)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_out, xgb_prob_out)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(16,9))
plt.subplot(1,2,1)

plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for XGBOOST OUT')
plt.legend(loc="lower right")
# plt.show()

# Extracting one tree from the Random Forest model
# single_tree = rf_model.estimators_[0]

# # Plotting the tree
# plt.figure(figsize=(20,10))
# plot_tree(single_tree, filled=True, feature_names=features, max_depth=3)
# plt.title("Decision Tree from Random Forest")
# plt.show()
precision, recall, _ = precision_recall_curve(y_out, xgb_prob_out)
pr_auc = average_precision_score(y_out, xgb_prob_out)

# Plot the Precision-Recall curve
plt.subplot(1,2,2)
plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (AUC = %0.2f)' % pr_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve OUT')
plt.legend(loc="lower right")
plt.show()
#%% Logistic Regression
logistic_model = LogisticRegression(random_state=42)
# Train and evaluate Logistic Regression
logistic_model.fit(X_train, y_train)
logistic_pred = logistic_model.predict(X_test)
logistic_accuracy = accuracy_score(y_test, logistic_pred)
logistic_class_rep = classification_report(y_test, logistic_pred)
logistic_confusion_matrix = confusion_matrix(y_test, logistic_pred)

logistic_prob = logistic_model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, logistic_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(16,9))
plt.subplot(1,2,1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Logistic Regression')
plt.legend(loc="lower right")
# plt.show()s
precision, recall, _ = precision_recall_curve(y_test, logistic_prob)
pr_auc = average_precision_score(y_test, logistic_prob)
plt.subplot(1,2,2)
plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (AUC = %0.2f)' % pr_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.show()

# Predict on the out of sample set
y_pred_out = logistic_model.predict(X_out)
logistic_prob_out = logistic_model.predict_proba(X_out)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_out, logistic_prob_out)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(16,9))
plt.subplot(1,2,1)

plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Logistic OUT')
plt.legend(loc="lower right")
# plt.show()

# Extracting one tree from the Random Forest model
# single_tree = rf_model.estimators_[0]

# # Plotting the tree
# plt.figure(figsize=(20,10))
# plot_tree(single_tree, filled=True, feature_names=features, max_depth=3)
# plt.title("Decision Tree from Random Forest")
# plt.show()
precision, recall, _ = precision_recall_curve(y_out, logistic_prob_out)
pr_auc = average_precision_score(y_out, logistic_prob_out)

# Plot the Precision-Recall curve
plt.subplot(1,2,2)
plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (AUC = %0.2f)' % pr_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve OUT')
plt.legend(loc="lower right")
plt.show()
# (xgb_accuracy, lgb_accuracy, logistic_accuracy), (xgb_class_rep, lgb_class_rep, logistic_class_rep)
#%%


