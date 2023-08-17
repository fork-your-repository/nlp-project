import pandas as pd
import numpy as np
import nltk.sentiment
import nltk
import re
from nltk.tokenize import ToktokTokenizer
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import env as e
import acquire as a
import os
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

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

def clean(text: str) -> list: 
    """
    Cleans up the input text data.

    Args:
    text (str): Input text.

    Returns:
    Cleaned list of words.

    """
    text = (text.encode('ascii', 'ignore')
                .decode('utf-8', 'ignore')
                .lower())
    words = re.sub(r'[^\w\s]', ' ', text).split()
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = set(nltk.corpus.stopwords.words('english'))
    return [wnl.lemmatize(word) for word in words if word not in stopwords]


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

def analyze_sentiment(train):
    """
    Analyzes sentiment distribution across different programming languages.

    Args:
    train (pandas.DataFrame): DataFrame containing the training data.

    Returns:
    sentiment_info (pandas.DataFrame): DataFrame with mean and median sentiment values.

    """
    # Group the data by 'language' and aggregate the 'sentiment' column to compute its mean and median values
    sentiment_info = train.groupby('language').sentiment.agg(['mean', 'median'])

    # Plot a Kernel Density Estimate (KDE) of the sentiment distribution for each programming language
    # 'common_norm=False' ensures that each language's distribution is plotted using its own peak, rather than a common one
    sns.kdeplot(data=train, x='sentiment', hue='language', common_norm=False)

    # Label the x-axis as 'Sentiment'
    plt.xlabel('Sentiment')

    # Label the y-axis as 'Density'
    plt.ylabel('Density')

    # Set the title of the plot
    plt.title('Sentiment Distribution by Language')

    # Display the legend to identify each language's distribution
    plt.legend()

    # Show the plot
    plt.show()

    # Return the DataFrame containing mean and median sentiment values for each language
    return sentiment_info

# Other functions remain the same

