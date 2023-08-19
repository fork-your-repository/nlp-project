########################### MODELING #######################################

def split_data(df,variable):
    """
    Splits the data into train, validate, and test DataFrames.

    Args:
    df (pandas.DataFrame): Input DataFrame.
    variable (str): Target variable name.

    Returns:
    train, validate, test DataFrames.
    """
    # Split data into train, validate, and test
    train_validate, test = train_test_split(df, test_size=0.20, random_state=123, stratify=df[variable])
    train, validate = train_test_split(train_validate, test_size=0.25, random_state=123, stratify=train_validate[variable])
    return train, validate, test


def prepare_for_modeling(train, validate, test):
    """
    Prepare the data for modeling by creating feature and target variables.

    Args:
    train (pandas.DataFrame): Training data.
    validate (pandas.DataFrame): Validation data.
    test (pandas.DataFrame): Test data.

    Returns:
    X_bow, X_validate_bow, X_test_bow, y_train, y_validate, y_test, feature_names
    """
    # Create feature and target variables
    X_train = train.extra_clean_contents
    X_validate = validate.extra_clean_contents
    X_test = test.extra_clean_contents
    y_train = train.language
    y_validate = validate.language
    y_test = test.language

    # Create bag-of-words representations
    cv = CountVectorizer()
    X_bow = cv.fit_transform(X_train)
    X_validate_bow = cv.transform(X_validate)
    X_test_bow = cv.transform(X_test)
    
    feature_names = cv.get_feature_names_out()
    
    return X_bow, X_validate_bow, X_test_bow, y_train, y_validate, y_test, feature_names


def decision_tree(X_bow, X_validate_bow, y_train, y_validate):
    """
    Train a decision tree classifier and evaluate performance.

    Args:
    X_bow, X_validate_bow: Bag-of-words representations.
    y_train, y_validate: Target variables.

    Returns:
    scores_df (pandas.DataFrame): Accuracy scores for different max_depth values.
    """
    # Train and evaluate decision tree classifier
    scores_all = []
    for x in range(1, 20):
        tree = DecisionTreeClassifier(max_depth=x, random_state=123)
        tree.fit(X_bow, y_train)
        train_acc = tree.score(X_bow, y_train)
        val_acc = tree.score(X_validate_bow, y_validate)
        score_diff = train_acc - val_acc
        scores_all.append([x, train_acc, val_acc, score_diff])

    scores_df = pd.DataFrame(scores_all, columns=['max_depth', 'train_acc', 'val_acc', 'score_diff'])

    # Visualize results
    # (visualization code here)

    return scores_df


def random_forest_scores(X_bow, y_train, X_validate_bow, y_validate):
    """
    Train and evaluate a random forest classifier with different hyperparameters.

    Args:
    X_bow, X_validate_bow: Bag-of-words representations.
    y_train, y_validate: Target variables.

    Returns:
    df (pandas.DataFrame): Model performance summary.
    """
    # Train and evaluate random forest classifier with different hyperparameters
    # (rest of the function's code)

    return df


def best_model_classification_matrix(X_bow, y_train, X_validate_bow, y_validate):
    """
    Train a decision tree classifier with the best hyperparameters and produce classification reports.

    Args:
    X_bow, X_validate_bow: Bag-of-words representations.
    y_train, y_validate: Target variables.

    Returns:
    None
    """
    # Train decision tree classifier with best hyperparameters
    # Produce classification reports
    # (rest of the function's code)
    pass  # Pass statement to indicate the end of the function
