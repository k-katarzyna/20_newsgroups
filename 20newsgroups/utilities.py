import os
import random
from operator import itemgetter
from typing import List, Tuple, Any, TypeVar

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd


T = TypeVar("T", bound="DataCleaner")


def display_random_sample(data: List[str],
                          target_variable: np.ndarray,
                          labels: List[str],
                          n_samples: int = 1,
                          target_class: str = None
                         ) -> List[int]:
    """Easy display random sample(s) from all the data or from a selected class"""
    if target_class is not None:
        filtered_indices = [i for i, label in enumerate(target_variable)
                            if label == target_class]
        if not filtered_indices:
            print(f"No samples found for class {target_class}")
            return
        data = list(itemgetter(*filtered_indices)(data))
        target_variable = list(itemgetter(*filtered_indices)(target_variable))
    
    sampled_indices = random.sample(range(len(data)), k=n_samples)
    
    for i in sampled_indices:
        label = labels[target_variable[i]]
        print(f"Label: {label}\n\nMessage:\n{data[i]}\n")

    return sampled_indices


def check_empty_samples(X_train: List[str],
                        X_test: List[str]
                       ) -> Tuple[np.ndarray]:
    """Check if there are any empty strings in data."""
    empty_train_indices = [idx for idx, string in enumerate(X_train)
                           if string in ["", " "]]
    empty_test_indices = [idx for idx, string in enumerate(X_test)
                          if string in ["", " "]]
    
    return empty_train_indices, empty_test_indices


def remove_empty_samples(X: List[str], y: np.ndarray, indices: List[int]
                        ) -> Tuple[List[str], np.ndarray]:
    """Remove empty string samples."""
    X, y = X.copy(), y.copy()
    for idx in reversed(indices):
        del X[idx]
        y = np.delete(y, idx)

    return X, y


class DataCleaner:
    """
    A class for cleaning and preprocessing textual data.

    Parameters:
    - data: List of strings containing text data.
    - to_remove_patterns: List of regex patterns to remove from data.
    - to_remove_words: List of specific words to remove from data.
    - token_pattern: Regex pattern to tokenize words.
    """
    def __init__(self,
                 data: List[str] = None,
                 to_remove_patterns: List[str] = [],
                 to_remove_words: List[str] = [],
                 token_pattern: str = r"(?u)\b\w[\w']*\b"
                ) -> None:
        """Initialize DataCleaner object."""
        self.data = data.copy() if data is not None else None
        self.to_remove_patterns = to_remove_patterns
        self.to_remove_words = to_remove_words
        self.token_pattern = token_pattern
        self.stop_words = stopwords.words("english")
        self.lemmatizer = WordNetLemmatizer()

    def remove_substrings_from_patterns(self: T) -> T:
        "Remove substrings found as unwanted token patterns."
        for pattern in self.to_remove_patterns:
            self.data = [pattern.sub("", message) for message in self.data]

        return self
    
    def remove_substrings_from_list(self: T) -> T:
        """Tokenize, remove unwanted words and lowercase all."""
        pre_cleaned = []
        
        for message in self.data:
            tokens = re.findall(self.token_pattern, message)
            cleaned_message = [w.lower() for w in tokens
                               if w.lower() not in self.stop_words + self.to_remove_words]
            pre_cleaned.append(cleaned_message)

        self.data = pre_cleaned
    
        return self       
    
    def lemmatize_data(self: T) -> T:
        """Lemmatize tokenized messages and join into strings."""
        for i in range(len(self.data)):
            lemmatized_message = [self.lemmatizer.lemmatize(token) for token in self.data[i]]
            self.data[i] = " ".join(lemmatized_message)
    
        return self

    def transform(self: T, data: List[str] = None) -> List[str]:
        """Transform data with cleaning and lemmatization"""
        if data is not None:
            self.data = data.copy()
        self.remove_substrings_from_patterns()
        self.remove_substrings_from_list()
        self.lemmatize_data()
        
        return self.data


def get_metrics(cv_results: dict,
                main_metric: str = None,
                other_metrics: List[str] = None
               ) -> dict:
    """Get metrics from `cv_results_` dict provided by RandomizedSearchCV optimizer."""
    metrics = {}
    if main_metric is None:
        main_metric = "f1_macro"
    if other_metrics is None:
        other_metrics = ["f1_weighted", "accuracy"]
    
    best_index = np.argmax(cv_results[f"mean_test_{main_metric}"])

    best_score = cv_results[f"mean_test_{main_metric}"][best_index]
    best_score_train = cv_results[f"mean_train_{main_metric}"][best_index]

    metrics[main_metric] = best_score
    metrics[f"{main_metric}_train"] = best_score_train

    for metric in other_metrics:
        corresponding_metric_test = cv_results[f"mean_test_{metric}"][best_index]
        corresponding_metric_train = cv_results[f"mean_train_{metric}"][best_index]
        metrics[metric] = corresponding_metric_test
        metrics[f"{metric}_train"] = corresponding_metric_train

    metrics["time[s]"] = np.round((cv_results["mean_fit_time"][best_index]
                                   + cv_results["mean_score_time"][best_index]).mean(), 2)

    return metrics


def load_results(endswith: str,
                 directory: str = "results"
                ) -> Tuple[List[Any], List[str]]:
    """Load objects from files with specified filename ending in the given directory."""
    object_list = []
    name_list = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(endswith):
                file_path = os.path.join(root, file)
                object_ = load(file_path)
                object_list.append(object_)
                
                subfolder = os.path.basename(root)
                file_prefix = file[:3]
                name = f"{subfolder}_{file_prefix}"
                name_list.append(name)

    return object_list, name_list