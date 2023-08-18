import pandas as pd
import numpy as np
import nltk.sentiment
import nltk
import re
from nltk.tokenize import ToktokTokenizer
# import env as e
# import acquire as a
import os
import json
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

####################################### ACQUIRE ##################################

# List of JSON file names
json_file_names = [
    'data1.json',
    'data3.json',
    'data4.json',
    'data5.json',
    'data6.json',
    'data7.json',
    'data8.json',
    'data9.json',
    'data10.json',
    'data11.json',
    'data12.json',
    'data13.json',
    'data14.json',
    'data15.json',
    'data16.json',
    'data17.json',
    'data18.json',
    'data19.json',
    'data20.json',
    'data21.json',
    'data22.json',
    'data23.json',
    'data24.json'
]

# Combine JSON data from multiple files into a single list
combined_data = []

for file_name in json_file_names:
    with open(file_name, "r") as json_file:
        data = json.load(json_file)
        combined_data.extend(data)

# Write the combined data to a new JSON file named "data2.json"
with open("data2.json", "w") as combined_json_file:
    json.dump(combined_data, combined_json_file, indent=1)

    
################################### PREPARE ##########################################   
def tokenize(text):
    """
    Tokenizes the words in the input string.
    """
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    return tokens

def clean(text: str) -> list: 
    """
    Cleans up the input text data.
    """
    text = (text.encode('ascii', 'ignore')
                .decode('utf-8', 'ignore')
                .lower())
    
    words = re.sub(r'[^\w\s]', ' ', text).split()
    
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    return [wnl.lemmatize(word) for word in words if word not in stopwords]

def nlp_wrangle():
    """
    Performs data wrangling for natural language processing (NLP) tasks.
    Returns a processed DataFrame for NLP analysis.
    """
    # Load data from JSON file
    df = pd.read_json('data2.json')
    
    # Tokenize and clean contents
    df['clean_contents'] = df.readme_contents.apply(tokenize).apply(' '.join)
    df['clean_contents'] = df.clean_contents.apply(clean).apply(' '.join)
    
     # Words to remove
    words_to_remove = ['http', 'com', '124', 'www','github', 'top', 'go','107', '0','1','2','3','4', '5', '6', '7', '8','9', 'md','p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'em', 'abbr', 'q','ins', 'del', 'dfn', 'kbd', 'pre', 'samp', 'var', 'br', 'div', 'a', 'img', 'param', 'ul','ol', 'li', 'dl', 'dt', 'dd']

    # Remove specific words from clean_contents
    for word in words_to_remove:
        df['clean_contents'] = df['clean_contents'].str.replace(word, '')

    # Add message_length and word_count columns
    df['message_length'] = df['clean_contents'].str.len()
    df['word_count'] = df.clean_contents.apply(clean).apply(len)

    # Keep only top languages and assign others to 'Other'
    languages_to_keep = ['JavaScript', 'Python', 'Java', 'TypeScript', 'HTML']
    df['language'] = np.where(df['language'].isin(languages_to_keep), df['language'], 'Other')

    # Filter DataFrame based on conditions
    df = df.loc[(df['word_count'] <= 10000) & (df['message_length'] <= 60000)]

    return df

def intersection_list():
    words_df = nlp_wrangle()
    readme_words_list = words_df.clean_contents.to_list()
    readme_words_list

    readme_words = []
    for list in readme_words_list:
        split_list = list.split()
        readme_words.append(split_list)

    words_list = []
    for _ in readme_words:
        for el in _:
            words_list.append(el)

    dictionary_words = pd.read_csv('/usr/share/dict/words', header=None)
    dictionary_words = dictionary_words.drop(index=[122337,122338])
    dictionary_words = dictionary_words.squeeze()
    intersect = set(words_list) & set(dictionary_words)
    intersect = sorted(intersect)
    return intersect

def extra_clean_column(words_df):
    extra_clean_article = []
    for i in words_df.index:
        article_words = words_df.clean_contents[i].split()
        extra_clean = set(intersect) & set(article_words)
        extra_clean = sorted(extra_clean)
        extra_clean = ' '.join(extra_clean)
        extra_clean_article.append(extra_clean)

    words_df = words_df.assign(extra_clean_contents = extra_clean_article) 
    return words_df

################################################ EXPLORE #############################################


def create_bar_chart(df, column_name, title):
    """
    Creates a horizontal bar chart for a categorical column.

    Args:
    df (pandas.DataFrame): Input DataFrame.
    column_name (str): Column to visualize.
    title (str): Chart title.

    Returns:
    None.

    """
    # Get the counts of each unique value in the specified column
    values = df[column_name].value_counts()

    # Extract the names of the unique values (categories) as a list
    labels = values.index.tolist()

    # Extract the counts of each unique value as a list
    sizes = values.tolist()

    # Create a horizontal bar chart
    # 'skyblue' is the chosen color for the bars, but this can be customized
    plt.barh(labels, sizes, color='skyblue')

    # Label the x-axis as 'Count'
    plt.xlabel('Count')

    # Label the y-axis using the column name
    plt.ylabel(column_name)

    # Set the title of the chart
    plt.title(title)

    # Invert the y-axis so that the category with the highest count is at the top
    plt.gca().invert_yaxis()

    # Display the chart
    plt.show()

# Usage example:
# create_bar_chart(df, 'Category', 'Distribution of Categories')



def analyze_word_frequency(train, column_name, num_words):
    """
    Analyzes word frequency in a specific column across different programming languages.

    Args:
    train (pandas.DataFrame): DataFrame containing the training data.
    column_name (str): Column to analyze.
    num_words (int): Number of words to display.

    Returns:
    word_counts (pandas.DataFrame): DataFrame with word counts.

    """
    # Combine words based on programming languages
    words_by_language = {
        language: clean(' '.join(train[train.language == language][column_name]))
        for language in train.language.unique()
    }

    # Calculate word frequencies
    word_freqs_by_language = {
        language: pd.Series(words).value_counts().head(num_words)
        for language, words in words_by_language.items()
    }

    # Create word counts DataFrame
    word_counts = pd.DataFrame(word_freqs_by_language).fillna(0).astype(int)

    return word_counts


################################################## MODELING #########################################

def split_data(df, variable):
    """
    Splits the data into train, validate, and test DataFrames.

    Args:
    df (pandas.DataFrame): Input DataFrame.
    variable (str): Target variable name.

    Returns:
    train, validate, test DataFrames.

    """
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
    X_bow, X_validate_bow, X_test_bow, y_train, y_validate, y_test
    """
    # Create feature and target variables
    X_train = train.clean_contents
    X_validate = validate.clean_contents
    X_test = test.clean_contents
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
    Train and evaluate a random forest classifier with different hyperparameters.

    Args:
    X_bow, X_validate_bow: Bag-of-words representations.
    y_train, y_validate: Target variables.

    Returns:
    df (pandas.DataFrame): Model performance summary.
    """
    # Define hyperparameters
    train_scores = []
    validate_scores = []
    min_samples_leaf_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    max_depth_values = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    # Train and evaluate random forest classifier
    for min_samples_leaf, max_depth in zip(min_samples_leaf_values, max_depth_values):
        rf = RandomForestClassifier(min_samples_leaf=min_samples_leaf, max_depth=max_depth, random_state=123)
        rf.fit(X_bow, y_train)
        train_score = rf.score(X_bow, y_train)
        validate_score = rf.score(X_validate_bow, y_validate)
        train_scores.append(train_score)
        validate_scores.append(validate_score)

    # Calculate differences between train and validation scores
    diff_scores = [train_score - validate_score for train_score, validate_score in zip(train_scores, validate_scores)]

    # Create summary DataFrame
    
    df = pd.DataFrame({
        'min_samples_leaf': min_samples_leaf_values,
        'max_depth': max_depth_values,
        'train_score': train_scores,
        'validate_score': validate_scores,
        'score_difference': diff_scores
    })
    
    # Visualize results
    sns.set_style('whitegrid')
    plt.plot(df['min_samples_leaf'], df['train_score'], label='Train score')
    plt.plot(df['min_samples_leaf'], df['validate_score'], label='Validation score')
#     plt.fill_between(df['train_score'], df['validate_score'], alpha=0.2, color='gray')
    plt.xlabel('Min Samples Leaf')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Accuracy vs Max Depth')
    plt.legend()
    plt.show()

    return df