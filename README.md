# Sentiment Analysis Using Machine Learning

## Project Overview

This project focuses on classifying emotions expressed in text comments using machine learning. It involves text preprocessing, feature extraction using TF-IDF, training classification models, and evaluating their performance.

## Objectives

- Preprocess text data by cleaning, tokenizing, removing stop words, and applying stemming.
- Convert text into numerical features using TF-IDF.
- Train and compare Naive Bayes and Support Vector Machine (SVM) classifiers.
- Evaluate model performance using classification metrics.
- Predict emotion labels for new text comments.

## Technologies Used

- Python
- Pandas
- NLTK
- Scikit-learn
- Jupyter Notebook

## Project Workflow

1. **Data Loading:** Load the dataset and inspect its structure, columns, and missing values.
2. **Text Preprocessing:** Convert text to lowercase, remove URLs, tokenize sentences, remove stop words, and apply stemming.
3. **Feature Extraction:** Use `TfidfVectorizer` to convert text into numerical feature vectors.
4. **Data Splitting:** Divide the dataset into training and testing sets.
5. **Model Training:** Train two classification algorithms:
   - Multinomial Naive Bayes
   - Linear Support Vector Machine (SVM)
6. **Model Evaluation:** Compare the classifiers using accuracy, precision, recall, F1-score, classification reports, and confusion matrices.
7. **Prediction:** Use the trained models to predict emotion labels for new text comments.

## Understanding TF-IDF

TF-IDF stands for Term Frequency–Inverse Document Frequency. It converts text into numerical vectors by assigning weights to words based on their frequency within a document and their distribution across the collection of documents.

These numerical features allow machine learning algorithms to identify patterns in text and classify emotions.

## Evaluation Metrics

- **Accuracy:** Measures the proportion of correctly classified comments.
- **Precision:** Measures how many predictions for a particular class are correct.
- **Recall:** Measures how many actual examples of a class are correctly identified.
- **F1-score:** Combines precision and recall into a single metric.
- **Classification Report:** Provides precision, recall, F1-score, and support for each class.
- **Confusion Matrix:** Shows correct predictions and misclassifications across emotion classes.

## Dataset

The project uses a dataset containing text comments and their corresponding emotion labels.

Ensure that the dataset is available at the file path specified in the notebook and that the text and target column names match those used in the code.

## How to Run

1. Clone or download this repository.
2. Open `SentimentAnalysis.ipynb` in Jupyter Notebook or Google Colab.
3. Install the required libraries:

   ```bash
   pip install pandas nltk scikit-learn jupyter
   ```

4. Load the dataset using the correct file path.
5. Run the notebook cells in order.
6. Review the evaluation metrics and compare the performance of Naive Bayes and SVM.

## Results

The notebook evaluates both models and generates performance metrics for comparison. The better-performing model should be selected based on the actual evaluation results, including F1-score when class distribution is imbalanced.

## Future Improvements

- Experiment with CountVectorizer and compare it with TF-IDF.
- Improve text preprocessing while preserving important negation words.
- Tune model parameters to improve classification performance.
- Explore additional machine learning algorithms.
- Evaluate performance on more diverse text samples.

## Project Structure

```text
Sentiment-Analysis/
├── SentimentAnalysis.ipynb
├── README.md
└── nlp_dataset.csv
```

*Note: The dataset filename and project structure should be updated to match the files in your repository.*
