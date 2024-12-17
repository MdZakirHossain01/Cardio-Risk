#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Step 1: Data Loading and Exploration

# Importing necessary libraries
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


# Loading the dataset
file_path = 'cardio_train.csv'  # Update this path to your dataset location
df = pd.read_csv(file_path, sep=';')  # Assuming delimiter is semicolon (;)


# In[2]:


# Displaying the first few rows of the dataset
print("Dataset Preview:\n", df.head())


# In[3]:


# Basic dataset information
print("\nDataset Information:")
df.info()


# In[4]:


# Checking for missing values
print("\nMissing Values:\n")
print(df.isnull().sum())

# Checking for duplicate rows
duplicates = df.duplicated().sum()
print(f"\nNumber of Duplicate Rows: {duplicates}")


# In[5]:


# Displaying descriptive statistics
print("\nDescriptive Statistics:\n")
print(df.describe())


# In[6]:


# Class distribution in the target variable
print("\nClass Distribution (Target Variable):\n")
print(df['cardio'].value_counts())


# # Step 2: Data Preprocessing

# In[7]:


# Dropping duplicate rows if any
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"\nDuplicates dropped: {duplicates}")


# In[8]:


# Handling missing values (if any exist)
# Assuming no missing values for now; otherwise, use appropriate imputation techniques
print("\nChecking Missing Values Again:\n")
print(df.isnull().sum())


# In[9]:


# Creating a new feature: BMI (Body Mass Index)
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
print("\nBMI Feature Added:\n", df[['weight', 'height', 'BMI']].head())


# In[10]:


# Outlier detection and handling (Example: Removing unrealistic BMI values)
bmi_upper_limit = 60  # Assuming BMI above 60 is unrealistic
df = df[df['BMI'] <= bmi_upper_limit]
print(f"\nData after removing BMI outliers (> {bmi_upper_limit}):\n", df.shape)


# In[11]:


# Encoding categorical variables (if applicable)
# For now, assuming all columns are numeric; modify here if you have categorical features

# Reordering and displaying updated dataset
print("\nUpdated Dataset Columns:\n", df.columns)

# Saving the preprocessed dataset for further steps
preprocessed_file_path = 'cardio_train_preprocessed.csv'
df.to_csv(preprocessed_file_path, index=False)
print(f"\nPreprocessed dataset saved to: {preprocessed_file_path}")


# # Step 3: Exploratory Data Analysis (EDA)

# In[12]:


# Step 3: Exploratory Data Analysis (EDA)
import matplotlib.pyplot as plt
import seaborn as sns

# Setting plot style
sns.set(style="whitegrid")

# 1. Distribution of the Target Variable
plt.figure(figsize=(6, 4))
sns.countplot(x='cardio', data=df, palette='viridis')
plt.title('Class Distribution of Cardiovascular Disease')
plt.xlabel('Cardio (0: No Disease, 1: Disease)')
plt.ylabel('Count')
plt.show()


# In[13]:


# 2. Correlation Heatmap
plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()


# In[14]:


# 3. Distribution of BMI
plt.figure(figsize=(8, 6))
sns.histplot(df['BMI'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of BMI')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.show()


# In[15]:


# 4. Age Distribution by Target Variable
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='age', hue='cardio', bins=30, kde=True, palette='muted', element='step')
plt.title('Age Distribution by Cardiovascular Disease')
plt.xlabel('Age (in Days)')
plt.ylabel('Frequency')
plt.show()


# In[17]:


# 5. Boxplot of BMI vs. Cardiovascular Disease
plt.figure(figsize=(8, 6))
sns.boxplot(x='cardio', y='BMI', data=df, palette='pastel')
plt.title('Boxplot of BMI by Cardiovascular Disease')
plt.xlabel('Cardio (0: No Disease, 1: Disease)')
plt.ylabel('BMI')
plt.show()


# # Step 4: Data Splitting and Normalization

# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Features and target variable
X = df.drop(columns=['cardio'])  # Dropping the target variable
y = df['cardio']  # Target variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nData Split Completed:")
print(f"Training Data Shape: {X_train.shape}, Training Labels Shape: {y_train.shape}")
print(f"Testing Data Shape: {X_test.shape}, Testing Labels Shape: {y_test.shape}")


# In[19]:


# Normalizing the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling Completed.")

# Converting scaled data back to DataFrame for clarity
X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

print("\nScaled Training Data Preview:\n", X_train.head())


# # Step 5: Model Building - Deep Learning Model

# In[22]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Defining the deep learning model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Training the model
history = model.fit(X_train, y_train, epochs=80, batch_size=32, validation_split=0.2, verbose=1)

# Evaluating the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")


# In[ ]:





# In[24]:

# Import TensorFlow layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Adding Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

# Updated Model
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification output
])

# Compile with a smaller learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train with callbacks
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, verbose=1,
                    callbacks=[early_stopping, lr_scheduler])

# Evaluate the updated model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"\nUpdated Test Loss: {loss:.4f}, Updated Test Accuracy: {accuracy:.4f}")


# In[ ]:





# In[28]:





# In[30]:


# Import necessary libraries
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Define base models with optimized hyperparameters
base_models = [
    ('xgb', XGBClassifier(
        n_estimators=300, 
        learning_rate=0.05, 
        max_depth=5, 
        objective='binary:logistic', 
        eval_metric='logloss', 
        use_label_encoder=False
    )),
    ('logreg', LogisticRegression(max_iter=1000, solver='liblinear'))
]

# Create Stacking Classifier
ensemble = StackingClassifier(
    estimators=base_models, 
    final_estimator=LogisticRegression(max_iter=1000, solver='liblinear')
)

# Train Ensemble Model
ensemble.fit(X_train, y_train)
ensemble_accuracy = ensemble.score(X_test, y_test)
print(f"\nImproved Ensemble Model Test Accuracy: {ensemble_accuracy:.4f}")


# In[34]:


dsd


# In[38]:


from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Update base models with reduced complexity
base_models = [
    ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, eval_metric='logloss')),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)),
    ('logreg', LogisticRegression(max_iter=1000, solver='liblinear'))
]


# Create Stacking Classifier
ensemble = StackingClassifier(
    estimators=base_models, 
    final_estimator=LogisticRegression(max_iter=2000, solver='liblinear')
)

# Train and Evaluate
ensemble.fit(X_train, y_train)
ensemble_accuracy = ensemble.score(X_test, y_test)
print(f"\nImproved Ensemble Model Test Accuracy: {ensemble_accuracy:.4f}")


# In[ ]:





# In[40]:


# Step 6: Optimized Ensemble Model

# Import necessary libraries
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Define base models with optimized hyperparameters
base_models = [
    ('xgb', XGBClassifier(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=5, 
        objective='binary:logistic', 
        eval_metric='logloss'
    )),
    ('rf', RandomForestClassifier(
        n_estimators=100, 
        max_depth=6, 
        random_state=42
    )),
    ('gb', GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=4, 
        random_state=42
    )),
    ('logreg', LogisticRegression(max_iter=1000, solver='liblinear'))
]

# Create and train the Stacking Classifier
ensemble = StackingClassifier(
    estimators=base_models, 
    final_estimator=LogisticRegression(max_iter=1000, solver='liblinear'),
    n_jobs=-1
)

# Train and Evaluate
ensemble.fit(X_train, y_train)
ensemble_accuracy = ensemble.score(X_test, y_test)

print(f"\nOptimized Ensemble Model Test Accuracy: {ensemble_accuracy:.4f}")


# In[ ]:





# In[41]:


# Step 6: Optimized Voting Classifier

# Import necessary libraries
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Define base models with optimized hyperparameters
base_models = [
    ('xgb', XGBClassifier(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=5, 
        objective='binary:logistic', 
        eval_metric='logloss'
    )),
    ('rf', RandomForestClassifier(
        n_estimators=100, 
        max_depth=6, 
        random_state=42
    )),
    ('gb', GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=4, 
        random_state=42
    )),
    ('logreg', LogisticRegression(max_iter=1000, solver='liblinear'))
]

# Create and train the Voting Classifier
voting_clf = VotingClassifier(
    estimators=base_models, 
    voting='soft',
    n_jobs=-1
)

# Train and Evaluate
voting_clf.fit(X_train, y_train)
voting_accuracy = voting_clf.score(X_test, y_test)

print(f"\nOptimized Voting Classifier Test Accuracy: {voting_accuracy:.4f}")


# In[ ]:





# In[42]:


# Step 7: Model Evaluation and Performance Metrics

# Importing necessary libraries
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Evaluate Voting Classifier
y_pred = voting_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, voting_clf.predict_proba(X_test)[:, 1])

# Print Evaluation Metrics
print(f"\nFinal Model Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, voting_clf.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# # Precision-Recall Curve

# In[43]:


from sklearn.metrics import precision_recall_curve

# Plot Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, voting_clf.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='green', label='Precision-Recall Curve')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()


# # Confusion Matrix Heatmap

# In[44]:


import seaborn as sns

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# # Feature Importance (XGBoost Only)

# In[45]:


# Feature Importance from XGBoost
xgb_model = voting_clf.named_estimators_['xgb']
plt.figure(figsize=(10, 8))
plt.barh(xgb_model.feature_importances_.argsort(), xgb_model.feature_importances_)
plt.yticks(range(len(X_train.columns)), X_train.columns[xgb_model.feature_importances_.argsort()])
plt.title('XGBoost Feature Importance')
plt.show()


# In[46]:


# Step 8: Model Deployment Preparation

# Importing required libraries
import joblib
import os

# Define model export path
model_export_path = 'voting_classifier_model.pkl'

# Save the trained Voting Classifier model
joblib.dump(voting_clf, model_export_path)
print(f"Model saved to {model_export_path}")

# Load the saved model for testing
loaded_model = joblib.load(model_export_path)
loaded_accuracy = loaded_model.score(X_test, y_test)
print(f"Loaded Model Test Accuracy: {loaded_accuracy:.4f}")

# Save important metadata (e.g., feature names) for deployment
metadata = {
    'features': list(X_train.columns),
    'model_export_path': model_export_path
}
metadata_export_path = 'model_metadata.pkl'
joblib.dump(metadata, metadata_export_path)
print(f"Metadata saved to {metadata_export_path}")


# In[50]:


# Step 9: Model Deployment - Streamlit Setup

# Importing required libraries
import streamlit as st
import joblib
import pandas as pd

# Load the saved model and metadata
model = joblib.load('voting_classifier_model.pkl')
metadata = joblib.load('model_metadata.pkl')

# Streamlit app setup
st.title("Cardiovascular Disease Risk Prediction")
st.write("Enter patient data to predict cardiovascular disease risk.")

# User input form
age = st.number_input("Age (days):", min_value=0, step=1)
height = st.number_input("Height (cm):", min_value=50, max_value=250, step=1)
weight = st.number_input("Weight (kg):", min_value=10, max_value=300, step=1)
bp_high = st.number_input("Systolic Blood Pressure:", min_value=50, max_value=250, step=1)
bp_low = st.number_input("Diastolic Blood Pressure:", min_value=30, max_value=150, step=1)
cholesterol = st.selectbox("Cholesterol Level:", options=[1, 2, 3], format_func=lambda x: f"Level {x}")
glucose = st.selectbox("Glucose Level:", options=[1, 2, 3], format_func=lambda x: f"Level {x}")
smoke = st.selectbox("Smoking Status:", options=[0, 1], format_func=lambda x: "Smoker" if x == 1 else "Non-Smoker")
alcohol = st.selectbox("Alcohol Intake:", options=[0, 1], format_func=lambda x: "Drinks Alcohol" if x == 1 else "Non-Drinker")
activity = st.selectbox("Physical Activity:", options=[0, 1], format_func=lambda x: "Active" if x == 1 else "Inactive")

# Prediction logic
if st.button("Predict"):
    input_data = pd.DataFrame([[age, height, weight, bp_high, bp_low, cholesterol, glucose, smoke, alcohol, activity]], 
                              columns=metadata['features'])
    prediction = model.predict(input_data)[0]
    prediction_prob = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")
    st.write("Prediction: ", "Disease Present" if prediction == 1 else "No Disease")
    st.write(f"Probability of No Disease: {prediction_prob[0]*100:.2f}%")
    st.write(f"Probability of Disease: {prediction_prob[1]*100:.2f}%")


# In[ ]:




