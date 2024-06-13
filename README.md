### Dataset Description:
The project utilizes the "20 Newsgroups" dataset, consisting of approximately 18 000 posts across 20 different topics. The split between the training and test sets is based on the dates of the posts, distinguishing those published before and after a specific cutoff date.

### Project Goal:
The aim of the project is to train a text classification model for all 20 categories with optimization focused on the F1 macro metric.

### Methods:
To achieve this goal, we employ:
- Classifiers: SVM (Support Vector Machine) and Multinomial Naive Bayes.
- We test 2 approaches to data preparation:
  1. Text cleaning and lemmatization.
  2. Unchanged text, preprocessed only by TfidfVectorizer.
- We evaluate 2 feature selection approaches:
  1. Chi-square test.
  2. Limiting the number of features through vectorization parameters.