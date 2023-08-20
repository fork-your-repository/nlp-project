# Imports Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from matplotlib.colors import LinearSegmentedColormap

# Suppressing warnings
import warnings
warnings.filterwarnings('ignore')




########################### MODELING #######################################


def split_data(df, variable):
    """
    Splits the data into train, validate, and test DataFrames based on the specified target variable.

    Args:
    df (pandas.DataFrame): Input DataFrame.
    variable (str): Target variable name.

    Returns:
    train, validate, test (pandas.DataFrame): Splitted DataFrames.
    """
    # Splitting the data into train_validate and test sets, then further splitting train_validate into train and validate sets
    train_validate, test = train_test_split(df, test_size=0.20, random_state=123, stratify=df[variable])
    train, validate = train_test_split(train_validate, test_size=0.25, random_state=123, stratify=train_validate[variable])
    
    return train, validate, test

def prepare_for_modeling(train, validate, test):
    """
    Prepares the input datasets for modeling. Extracts features and target variables and 
    converts text data into bag-of-words representation.

    Args:
    train, validate, test (pandas.DataFrame): DataFrames containing training, validation, and test data respectively.

    Returns:
    X_bow, X_validate_bow, X_test_bow (sparse matrix): Bag-of-words representations of the input datasets.
    y_train, y_validate, y_test (pandas.Series): Target variables for the input datasets.
    feature_names (list): Feature names for the bag-of-words representation.
    """
    # Extracting features and target variables from the data
    X_train = train.extra_clean_contents
    X_validate = validate.extra_clean_contents
    X_test = test.extra_clean_contents
    y_train = train.language
    y_validate = validate.language
    y_test = test.language

    # Creating bag-of-words representations for the features using CountVectorizer
    cv = CountVectorizer()
    X_bow = cv.fit_transform(X_train)
    X_validate_bow = cv.transform(X_validate)
    X_test_bow = cv.transform(X_test)
    
    feature_names = cv.get_feature_names_out()
    
    return X_bow, X_validate_bow, X_test_bow, y_train, y_validate, y_test, feature_names


def decision_tree(X_bow, X_validate_bow, y_train, y_validate):
    """
    Train a decision tree classifier for various depths and evaluate its performance 
    on both training and validation datasets.

    Args:
    X_bow, X_validate_bow: Bag-of-words representations of training and validation datasets.
    y_train, y_validate: Target variables for training and validation datasets.

    Returns:
    scores_df (pandas.DataFrame): DataFrame containing accuracy scores for different max_depth values.
    """
    # Training decision tree classifier for different depths and evaluating its performance
    scores_all = []
    for x in range(1, 20):
        tree = DecisionTreeClassifier(max_depth=x, random_state=123)
        tree.fit(X_bow, y_train)
        train_acc = tree.score(X_bow, y_train)
        val_acc = tree.score(X_validate_bow, y_validate)
        score_diff = train_acc - val_acc
        scores_all.append([x, train_acc, val_acc, score_diff])

    scores_df = pd.DataFrame(scores_all, columns=['max_depth', 'train_acc', 'val_acc', 'score_diff'])

    # Visualizing the accuracy scores for different max_depth values
    sns.set_style('whitegrid')
    plt.plot(scores_df['max_depth'], scores_df['train_acc'], label='Train score')
    plt.plot(scores_df['max_depth'], scores_df['val_acc'], label='Validation score')
    plt.fill_between(scores_df['max_depth'], scores_df['train_acc'], scores_df['val_acc'], alpha=0.2, color='gray')
    plt.xlabel('Max depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Accuracy vs Max Depth')
    plt.legend()
    plt.show()

    return scores_df

def random_forest_scores(X_bow, y_train, X_validate_bow, y_validate):
    """
    Train and evaluate a random forest classifier with varying hyperparameters. 
    Visualizes the accuracy scores for different hyperparameter combinations.

    Args:
    X_bow, X_validate_bow: Bag-of-words representations of training and validation datasets.
    y_train, y_validate: Target variables for training and validation datasets.

    Returns:
    df (pandas.DataFrame): DataFrame containing model performance metrics for different hyperparameter combinations.
    """
    # Defining hyperparameters for the random forest classifier
    train_scores = []
    validate_scores = []
    min_samples_leaf_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    max_depth_values = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    # Training random forest classifier with different hyperparameters and evaluating its performance
    for min_samples_leaf, max_depth in zip(min_samples_leaf_values, max_depth_values):
        rf = RandomForestClassifier(min_samples_leaf=min_samples_leaf, max_depth=max_depth, random_state=123)
        rf.fit(X_bow, y_train)
        train_score = rf.score(X_bow, y_train)
        validate_score = rf.score(X_validate_bow, y_validate)
        train_scores.append(train_score)
        validate_scores.append(validate_score)

    # Calculating differences between train and validation accuracy scores
    diff_scores = [train_score - validate_score for train_score, validate_score in zip(train_scores, validate_scores)]
    
    # Creating a summary DataFrame for the results
    df = pd.DataFrame({
        'min_samples_leaf': min_samples_leaf_values,
        'max_depth': max_depth_values,
        'train_score': train_scores,
        'validate_score': validate_scores,
        'score_difference': diff_scores
    })
    
    # Visualizing the accuracy scores for different hyperparameter combinations
    sns.set_style('whitegrid')
    plt.plot(df['min_samples_leaf'], df['train_score'], label='Train score')
    plt.plot(df['min_samples_leaf'], df['validate_score'], label='Validation score')
    plt.xlabel('Min Samples Leaf')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Accuracy vs Max Depth')
    plt.legend()
    plt.show()

    return df
def k_nearest(X_bow, y_train, X_validate_bow, y_validate):
    """
    Trains and evaluates KNN models for different values of k and plots the results.

    Parameters:
    -----------
    X_bow: array-like, shape (n_samples, n_features)
        Bag-of-words representations of training samples.
    y_train: array-like, shape (n_samples,)
        Target values for the training samples.
    X_validate_bow: array-like, shape (n_samples, n_features)
        Bag-of-words representations of validation samples.
    y_validate: array-like, shape (n_samples,)
        Target values for the validation samples.

    Returns:
    --------
    results: pandas DataFrame
        Contains the train and validation accuracy for each value of k.
    """
    # Initializing lists to store various metrics for different values of k
    metrics = []
    train_score = []
    validate_score = []

    # Iterating through different values of k to train and evaluate the KNN model
    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_bow, y_train)
        train_score.append(knn.score(X_bow, y_train))
        validate_score.append(knn.score(X_validate_bow, y_validate))
        diff_score = train_score[-1] - validate_score[-1]
        metrics.append({'k': k, 'train_score': train_score[-1], 'validate_score': validate_score[-1], 'diff_score': diff_score})

    # Computing baseline accuracy for comparison
    baseline_accuracy = (y_train == 6).mean()

    # Storing metrics in a pandas DataFrame
    results = pd.DataFrame.from_records(metrics)

    # Visualizing the results using a plot
    sns.set_style('whitegrid')
    plt.plot(results['k'], results['train_score'], label='Train score')
    plt.plot(results['k'], results['validate_score'], label='Validation score')
    plt.xlabel('K Neighbors')
    plt.ylabel('Accuracy')
    plt.title('KNN')
    plt.legend()
    plt.show()

    return results


def best_model_classification_matrix(X_bow, y_train, X_validate_bow, y_validate):
    """
    Trains a decision tree with a maximum depth of 1 and provides classification reports for train and validation sets.

    Parameters:
    -----------
    X_bow: array-like, shape (n_samples, n_features)
        Bag-of-words representations of training samples.
    y_train: array-like, shape (n_samples,)
        Target values for the training samples.
    X_validate_bow: array-like, shape (n_samples, n_features)
        Bag-of-words representations of validation samples.
    y_validate: array-like, shape (n_samples,)
        Target values for the validation samples.
    """
    # List to store training and validation accuracy and their differences
    scores_all = []

    # Training the decision tree model with a maximum depth of 1
    tree = DecisionTreeClassifier(max_depth=1, random_state=123)
    tree.fit(X_bow, y_train)
    train_acc = tree.score(X_bow, y_train)
    val_acc = tree.score(X_validate_bow, y_validate)
    score_diff = train_acc - val_acc
    scores_all.append([train_acc, val_acc, score_diff])

    # Getting predictions for the training data
    y_predictions = tree.predict(X_bow)
    
    # Displaying the classification report for training data
    report = classification_report(y_train, y_predictions, output_dict=True)
    print(f"Tree with max depth of 1 train")
    print(pd.DataFrame(report))
    print()

    # Getting predictions for the validation data
    y_validate_predictions = tree.predict(X_validate_bow)
    
    # Displaying the classification report for validation data
    report = classification_report(y_validate, y_validate_predictions, output_dict=True)
    print(f"Tree with max depth of 1 validate")
    print(pd.DataFrame(report))
    print()


def final_test(X_bow, y_train, X_validate_bow, y_validate, X_test_bow, y_test):
    """
    Trains a decision tree with a maximum depth of 1 on training data and evaluates its performance on test data.

    Parameters:
    -----------
    X_bow: array-like, shape (n_samples, n_features)
        Bag-of-words representations of training samples.
    y_train: array-like, shape (n_samples,)
        Target values for the training samples.
    X_validate_bow: array-like, shape (n_samples, n_features)
        Bag-of-words representations of validation samples.
    y_validate: array-like, shape (n_samples,)
        Target values for the validation samples.
    X_test_bow: array-like, shape (n_samples, n_features)
        Bag-of-words representations of test samples.
    y_test: array-like, shape (n_samples,)
        Target values for the test samples.
    """
    # List to store various metrics
    scores_all = []

    # Training the decision tree model with a maximum depth of 1
    tree = DecisionTreeClassifier(max_depth=1, random_state=123)
    tree.fit(X_bow, y_train)
    train_acc = tree.score(X_bow, y_train)
    test_acc = tree.score(X_test_bow, y_test)

    # Getting predictions for the test data
    y_test_predictions = tree.predict(X_test_bow)
    
    # Displaying the classification report for test data
    report = classification_report(y_test, y_test_predictions, output_dict=True)
    print(f"Tree with max depth of 1 train")
    print(pd.DataFrame(report))
    print(f'Test accuracy: {test_acc}')
