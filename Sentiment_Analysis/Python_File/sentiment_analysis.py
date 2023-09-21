# Importing Neccesary Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Regular Expression
import re

# Natural Language ToolKit
import nltk
from nltk.corpus import stopwords

# SciKit-Learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Importing the DATASET from my GitHub Repository itself (with raw-link)

url ="https://github.com/yashwanth-yadav-0103/Machine_Learning/raw/main/Sentiment_Analysis/tweets_covid19_dataset.csv"
df =pd.read_csv(url)

print("The size of the dataset is \n {}".format(df.shape))

# DATA PREPROCESSING....

# print(df.columns.tolist())

# print(df.head(3))

# print(df.nunique())

# print(df.isnull().sum())

# FEATURES AND LABELS....

# The input / Independent variable
feature = df.iloc[:,0].values

# The output / Dependent variable
label = df.iloc[:,1].values

# Data Processing with REGULAR EXPRESSION....

Filtered_data = []
for i in range(0, len(feature)):

    # To remove speacial characters
    Partial_Data = re.sub(r'\W', ' ', str(feature[i]))

    # To remove single characters
    Partial_Data= re.sub(r'\s+[a-zA-Z]\s+', ' ', Partial_Data)

    # To remove multiple spaces
    Partial_Data = re.sub(r'\s+', ' ', Partial_Data, flags=re.I)

    # To remove lower case
    Partial_Data = Partial_Data.lower()

    # Removing prefixed 'b'
    Partial_Data = re.sub(r'^b\s+', '', Partial_Data)

    # To remove single characters from the start
    Partial_Data = re.sub(r'\^[a-zA-Z]\s+', ' ', Partial_Data)


    Filtered_data.append(Partial_Data)

# Removing STOPWORDS and Performing Feature Extraction ( Converting Preprocessed Text Data into Numerical Data. )

nltk.download('stopwords')

vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))

Final_data= vectorizer.fit_transform(Filtered_data).toarray()

# print(Final_data)

# Spliting DATASET into Train Data and Test Data........

X_train, X_test, y_train, y_test = train_test_split(Final_data, label, test_size=0.15, random_state=0)

# Training the Model with Test Data using ML Algorithm (RFC)

Model= RandomForestClassifier(n_estimators=200, random_state=0)
Model.fit(X_train, y_train)

# Note: The Dataset is huge, Hence it might take more time while training the model with Train Dataset..



# Evaluating the Model... (Accuracy of the Model)

predict_sentiment = Model.predict(X_test)
Model_accuracy=(accuracy_score(y_test, predict_sentiment))*100

print("The Model Accuracy is {:.4f} % ".format(Model_accuracy)) 


# OUTPUT: We got Model accuracy of 95.5521 % 
