![Most Forked NLP](Most%20Forked%20NLP.jpg)

# <a name="top"></a>Most Forked Repositories Project- Forking Solutions
![]()


***
[[Project Description](#project_description)]
[[Project Goal](#project_goal)]
[[Initial Thoughts/Questions](#initial_thoughts_questions)]
[[Data Dictionary](#dictionary)]
[[Project Planning](#planning)]
[[Key Findings](#findings)]
[[Statistical Analysis](#stats)]
[[Modeling](#model)]
[[Conclusion](#conclusion)]
___



## <a name="project_description"></a>Project Description:

In this project we will be utilizing data from GitHub README files to analyze and predict programming language based on the content.
[[Back to top](#top)]

 ## <a name="project_goal"></a>Project Goal:

 The goal is to analyze develop a ML model that can accuratly  predict the primary programming language of a repository. This will permit higher effecacy categorization and organization of code repositories, making it easier for developers to find relevant projects and collaborate with others.
 [[Back to top](#top)]
 
 
## <a name="initial_thoughts_questions"></a>Initial Thoughts/Questions:
1. Are there any notable variations in the frequency of words between README files written in different programming languages?
2. Does the presence of specific libraries in the README file correspond with the programming language used?
3. What are the most frequently used words throughout the dataset and for each language?
4. What are the least frequently used words throughout the dataset and for each language?

 [[Back to top](#top)]

## <a name="dictionary"></a>Data Dictionary  
[[Back to top](#top)]


### Data Used
---
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
|repo| Name of User/Name of Repository on Github |float|
|language| The programming language used in the repository |object|
|readme_contents| The full contents of the README.md |object|
|clean_contents| Cleaned and lemmitzed form of the readme_contents |object|
|message_length| Length of the README by string length |int64|
|word_count| Length of the README by word count |int64|
|extra_clean_contents| List of words corresponding to actual dictionary words that appear in the MACBOOK user dictionary |object|

**

***

## <a name="planning"></a>Project Plan: 
[[Back to top](#top)]
- Acquire:
    - Acquired the data from github.com by extracting the ["Most Forked Repositories"]([https://www.kaggle.com/datasets/meirnizri/covid19-dataset](https://github.com/search?o=desc&q=stars:%3E1&s=forks&type=Repositories))
   
- The data was collected on August 17, 2023.
- The data was scraped and saved in a JSON file called data2.json.
- The JSON file contains information about github repositories.
- Prepare:
- Removed non-ASCII characters and converted all characters to lowercase. This was done to standardize the data and make it easier to process.
- Removed stopwords, tokenized, and lemmatized rows. Stopwords are words that are common and do not add much meaning to the text. Tokenization is the process of breaking the text into individual words. Lemmatization is the process of reducing words to their root form.
- Created a new column with cleaned and lemmatized README content. This was done to make the data more accessible for analysis.
- Created a bucket named 'other' to include all other languages that are not JavaScript, Python, Java, TypeScript, or HTML. This was done to group the less common languages together.
- Deleted extra words that were not relevant to the project. This was done to remove any words that were not relevant to the task of predicting the programming language.
- Split the data into train, validation, and test sets for exploration. This was done to ensure that the data was used fairly and that the results of the analysis were reliable.
  
- Exploration:
- Create data visualizations and answer the following questions:
 1. Are there any notable variations in the frequency of words between README files written in different programming languages?
2. Does the presence of specific libraries in the README file correspond with the programming language used?
3. What are the most frequently used words throughout the dataset and for each language?
4. What are the least frequently used words throughout the dataset and for each language?

- Modeling:
  - After converting the words into vectors, we will use accuracy as our evaluation metric. This means that we will measure the performance of our model by calculating the percentage of predictions that are correct.
- The baseline accuracy is 47.1%. This means that a model that predicts all languages randomly would be correct 47.1% of the time.
- We employed Decision Tree Classifier, Random Forest, and K-Nearest Neighbor as our models for predicting programming languages based on README content. These are three different machine learning algorithms that can be used to predict the programming language from the README content.
***


### Target Programming Language



*********************

# Modeling:

### Decision tree

- The best performing model was the decision tree with a max depth of 1. This model achieved an accuracy of 0.4 on the test set, which is slightly below baseline. However, it is important to note that the data was not perfectly clean, so this is a promising result. - Decision tree with a max depth of 1 is the best performing model. This model should be further evaluated with a cleaned dataset to see if it can achieve better accuracy.The model was more stable than the random forest and KNN models. The train/validate spread for the decision tree model was relatively consistent, while the train/validate spread for the random forest and KNN models was more erratic.
- The decision tree model was more interpretable than the random forest and KNN models. The decision tree model can be used to understand how the model makes predictions, while the random forest and KNN models are more black-box models.

### Random forest

- The random forest model with a min_sample_leaf and max_depth of 5 and 6, respectively, also performed well, with an accuracy of 0.41666 on the test set. However, this model was not significantly better than the decision tree model.

### K-nearest neighbors (KNN)

- The KNN model with 7 neighbors achieved an accuracy of 0.61666 on the test set, which is significantly better than the other models. However, this model is more sensitive to the dimensionality of the data, and the bag-of-words representation used in this experiment has 1167 dimensions. This is a high number of dimensions, and it can lead to overfitting.

- Overall, the decision tree model is the best performing model and is the most interpretable model. This model should be further evaluated with a cleaned dataset to see if it can achieve better accuracy.


# Conclusion

- The results were not as promising as due to an error in the cleaning process. However, the decision tree was still able to identify the most popular programming languages used in GitHub READMEs, even though the data was not perfectly clean. This suggests that there is enough information in the data to achieve baseline performance, even with some errors.
- These results indicate that there is still more work to be done to improve the cleaning pipeline and the analysis and modeling process. This is an opportunity to learn from mistakes and improve the overall process.


# Recommendations
- The cleaning pipeline should be more robust and should be able to handle a wider variety of errors.
- The analysis should be more thorough and should explore more features of the data.
- The models should be more sophisticated and should be able to learn from the data more effectively.
- By addressing these issues, we can improve the accuracy and performance of the system and make it more useful for developers.


# Takaways

- A proper acquire and prepare step is essential for any machine learning project.
- An error in cleaning the data can lead to inaccurate analysis and models.
- It is important to fix programmatic errors as soon as possible.
- Creating a list of all occurrences of all words before lemmatization will ensure that the data is clean and accurate.
- Doing so will allow for a proper TF-IDF feature extraction, which will likely take full advantage of the models presented.


# Next Steps

- Fix the programmatic error that resulted in the set of words post-lemmatize.
- Create a list of all occurrences of all words before lemmatizing.
- Re-run the analysis and models with the cleaned data.
- Evaluate the results of the new analysis and models.
- Make any necessary adjustments to the models.
- Continue to iterate on the project until the desired results are achieved.


# Steps on How to Reproduce Project:
1. Go to the nlp-project repository on GitHub.
2. Download the entire repository to your computer. You can do this by clicking on the "Code" button and selecting "Download ZIP". You can also copy the SSH code to your terminal and use that to clone the repository.
3. Generate a personal access token on GitHub. Go to https://github.com/settings/tokens and click on the "Generate new token" button. Make sure to leave all checkboxes unchecked to avoid selecting any scopes.
4. Create a file called env.py in the nlp-project directory on your computer.
5. Copy the generated personal access token and paste it into your env.py file under the variable github_token.
6. Similarly, add your GitHub username to your env.py file under the variable github_username.
7.  you have saved all the necessary information in your env.py file, you can run the final notebook.
