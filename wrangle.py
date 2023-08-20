
# Import Libraries
import pandas as pd
import numpy as np
import nltk.sentiment
import nltk
import re
from nltk.tokenize import ToktokTokenizer
import os
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Suppressing warnings
import warnings
warnings.filterwarnings('ignore')

####################################### ACQUIRE ##################################

def combine_json_files_and_save(json_file_names, output_file_name):
    combined_data = []

    for file_name in json_file_names:
        with open(file_name, "r") as json_file:
            data = json.load(json_file)
            combined_data.extend(data)

    with open(output_file_name, "w") as combined_json_file:
        json.dump(combined_data, combined_json_file, indent=1)

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


# Output file name for combined data
output_file_name = 'data2.json'

# Call the function to combine JSON files and write the output
combine_json_files_and_save(json_file_names, 'data2.json')


    
################################### PREPARE ##########################################   

# Initialize the tokenizer
def tokenize(text):
    """
    Tokenizes the words in the input string.
    """
    tokenizer = ToktokTokenizer()
    # Tokenize the input text
    tokens = tokenizer.tokenize(text)
    return tokens

# Convert to lowercase and handle encoding
def clean(text: str) -> list: 
    """
    Cleans up the input text data.
    """
    text = (text.encode('ascii', 'ignore')
                .decode('utf-8', 'ignore')
                .lower()) 
   # Remove non-word characters and split into words  
    words = re.sub(r'[^\w\s]', ' ', text).split()
    # Initialize WordNet lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    # Get a set of English stopwords
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

# Get the processed DataFrame from nlp_wrangle()
def intersection_list():
    words_df = nlp_wrangle()
    readme_words_list = words_df.clean_contents.to_list()
    readme_words_list
    
    # Convert cleaned contents to a list of lists (split list, create set intersection, return result)
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
    
    # Apply extra cleaning, create new column, return modified DataFrame)

# def extra_clean_column(words_df):
#     extra_clean_article = []
#     for i in words_df.index:
#         article_words = words_df.clean_contents[i].split()
#         extra_clean = set(intersect) & set(article_words)
#         extra_clean = sorted(extra_clean)
#         extra_clean = ' '.join(extra_clean)
#         extra_clean_article.append(extra_clean)

#     words_df = words_df.assign(extra_clean_contents = extra_clean_article) 
#     return words_df

def extra_clean_column(words_df, intersect):
    extra_clean_article = []
    for i in words_df.index:
        article_words = words_df.clean_contents[i].split()
        extra_clean = set(intersect) & set(article_words)
        extra_clean = sorted(extra_clean)
        extra_clean = ' '.join(extra_clean)
        extra_clean_article.append(extra_clean)

    words_df = words_df.assign(extra_clean_contents=extra_clean_article)
    return words_df


################################################ EXPLORE #############################################

def plot_ombre_bars(df, bar_height=0.5):
    """
    Plots a horizontal bar chart with an ombre color effect.

    Parameters:
    - df: DataFrame containing the data to be plotted (expects mean values to be computed).
    - bar_height: Thickness of each bar. Lower values result in thinner bars.

    Returns:
    - Plots the horizontal bar chart.
    """
    
    # Setting color palette for the ombre effect
    colors = ['darkred', 'red', 'darkorange','grey','black']
    cmap = LinearSegmentedColormap.from_list("ombre", colors)

    # Set the figure background color to black
    plt.figure(figsize=(10,7), facecolor='black')
    means = df.mean()

    # Plotting each bar with gradient colors
    for i, (index, value) in enumerate(means.iteritems()):
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        gradient = np.vstack((gradient, gradient))
        extent = [0, value, i + 1 - bar_height, i + 1]
        plt.imshow(gradient, aspect='auto', cmap=cmap, extent=extent)
    
    plt.yticks(np.arange(0.5, len(means) + 0.5), means.index)
    plt.xlim(0, means.max() + 10)  # Adjusting xlim for better visuals

    # Set the title, axes labels, and title colors to white
    plt.title('Bar Chart of All Languages', color='white')
    plt.ylabel('Programming Languages', color='white')
    plt.xlabel('Mean Value', color='white')
    plt.tick_params(axis='both', colors='white')
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.show()


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

# PREAPARE FOR MODELING