# CODSOFT
Project Overview
Objective: To build a model that can predict the genre of a movie based on its description.
Dataset: The dataset includes two files: train_data.txt and test_data.txt, containing the respective movie descriptions and genres.
Tools and Libraries:
Python libraries: numpy, pandas, matplotlib, seaborn, nltk, and scikit-learn.
Data preprocessing techniques, including text cleaning and TF-IDF vectorization, to prepare the data for modeling.
A Naive Bayes classifier is used to perform the classification task.
Key Steps
Data Loading: The dataset is loaded using Pandas.
Data Preprocessing:
Removal of unnecessary columns (ID and TITLE).
Checking for null values and duplicates.
Cleaning the text data by removing links, special characters, and stop words.
Exploratory Data Analysis: Visualization of genre distribution to understand the dataset better.
Feature Extraction: Use of TF-IDF vectorization to convert textual data into numerical format for model training.
Model Training: A Multinomial Naive Bayes model is trained on the processed data.
Model Evaluation: The model's performance is assessed using accuracy metrics on both the test set and external validation data.
Results
The model achieved a test accuracy of approximately 52.4%. However, there was a mismatch in the number of samples between the predicted labels and the true labels in the external test dataset, indicating a potential area for further refinement.

Future Work
Explore different machine learning models to improve classification accuracy.
Conduct hyperparameter tuning for better performance.
Experiment with additional NLP techniques such as word embeddings or deep learning approaches.
