## The content
The progress, results and comments are documented in the *20newsgroups/20newsgroups.ipynb* notebook, as well as summary and conclusions. Utilities designed for this project are located in *20newsgroups/utilities.py* file.

Use nbviewer to see the fully-rendered notebook with gradient styled result dataframes. [LINK](https://nbviewer.org/github/k-katarzyna/20_newsgroups/blob/main/20newsgroups/20newsgroups.ipynb)

## The dataset
The project utilizes the "20 Newsgroups" dataset, consisting over 18 000 posts across 20 different topics. The split between the training and test sets is based on the dates of the posts, distinguishing those published before and after a specific cutoff date.

## The goal
The aim of the project is to train a text classification model for all 20 categories with optimization focused on the F1 macro metric.

## Methods
To achieve the goal, we employed:
- Classifiers: SVM (Support Vector Machine) and Multinomial Naive Bayes.
- We tested 2 approaches to data preparation:
  * Text cleaning and lemmatization.
  * Unchanged text, preprocessed only by TfidfVectorizer.
- We evaluated 2 feature selection approaches:
  * Chi-square test.
  * Limiting the number of features through vectorization parameters.