# Machine Learning In Practice

Group members: Raia Abu Ahmad, Thawab Alkhiami.

This is a documentation file for the design decisions that were made during the development of the project.

We will explain the decisions made per ML pipeline steps. Namely, we will justify why we chose the specific steps that were used in our pipline.

# Preprocessing
Below, we will explain each step taken in preprocessing, and why it was done:

- ``create labels``: Create labels of TRUE (viral) or FALSE (not viral) for each tweet in the dataset, based on the like and retweet counts.
                     We did not change the default setting of weights = 1; and threshold = 50. 
                     
- `` hashtags and mentions remover``: Remove all text that comes after a hashtag or a mention. We decided to add this step based on the preliminary n-gram results we got (will be explained in detail below under 'n-grams').
                     
- ``punctuation remover``: Remove all punctuation signs from tweets. The rationale is that for further preprocessing steps, there is no need for punctuation to be part of the tweet text, 
                           as it will not add further meaning that can be used to predict the virality of a tweet. If we think that a certain algorithm benefits from the existence of punctuation,
                           we use the raw tweets instead (e.g., in sentiment analysis as will be explained below).
                           A few tweaks were done on the original algorithm given to us. Namely, adding the character '±' as a punctuation to remove, as well as removing whitespaces after removing punctuation.

- ``emoji splitter``: Add spaces before and after each emoji from a tweet. We implemented this because we want each emoji to be taken as its own token, and not part of a series of tokens, and since it is very common for people on social media posts not to
                      insert spaces between emojis, or sometimes between alphanumeric characters and emojis, we decided that this step is needed.
                      E.g., we want 'cute cats🐱🐈' to be tokenized as 'cute', 'cats', '🐱', '🐈'; and not 'cute', 'cats🐱🐈'. Thus 'cats' can then be counted correcctly as an occurence of the lemma 'cat'.
                   
- ``stop words remover``: Remove stop words from tweets. This is used as a precursor step to tokenization and lemmatization, which are then taken to extract N-grams.
                          In addition to this step being a common practice in most NLP pipelines, intuitvely, we don't think that stop words will add any value to the semantic aspect of the tweet, 
                          which we intend to check in later steps. This is why we decided to remove them, and focus on content words.
                          
- ``tokenizer``: Receive the tweets after removing punctuation and stop words, and tokenize them, i.e., segment them to separate tokens. This step is done as a precursor to the lemmatization step, 
                 as the lemmatizer takes as input a list of tokens, rather than a string.
                 
- ``lemmatizer``: Lemmatize each token in each tweet, i.e., replace the token with its lemma, which is the basic dictionary form of a word. 
                  We applied lemmatization instead of stemming because we think that lemmatization, though more computationally expensive
                  (but in our case it doesn’t seem to take long at all to process), we would like the information of certain verb and and noun inflections to be retained and processed in further steps as the same. E.g., when we work on n-grams, we want ‘high school’ and ‘high schools’ to be counted in the same frequency count. 
                  We know this can be done using a stemmer, but we have used a lemmatizer because we believe it will yield more specific results in that sense, and cover more cases. E.g., we want 'ran a marathon' and 'run a marathon' to be counted as the same, because thay have exactly the same semantic sense. 
                  It is important to note, that the unit test showed us that the nltk WordNetLemmatizer is not perfect; e.g. not converting ‘heard’ to ‘hear’ or ‘triggered’ to ‘trigger’. Optimally, we would use another lemmatizer (maybe from spaCy) but for time constraints on this project we kept working with nltk. 
                  

- ``ngrams``: There are several decisions that had to be taken when it came to extracting n-grams and translating them as features in our pipline.
              Below, we will explain the rationale behind each decision that was taken:
              
- Choosing bigrams over other n-grams: We were not entirely sure which n in n-grams should be used. But, after inspecting the dataset we have, since it contains tweets that have to do with data science, we concluded that it would be best to check for bigrams.
              This is because a lot of keywords in that field are collocations of two words; e.g. 'Artificial Intelligence', 'Machine Learning', 'Deepl Learning', 'Big Data', and so on.
              To back this up, we tried to compare the 30 most frequent bigrams to the 30 most frequenct trigrams, which can be seen in the plots below:
              
 1. 30 most frequent bigrams:             
<img src="https://github.com/ryabhmd/MLinPractice/blob/main/images/30_freq_bigrams_with_hashtags.png" />
       
 2. 30 most frequent trigrams:
 <img src="https://github.com/ryabhmd/MLinPractice/blob/main/images/30_freq_trigrams_with_hashtags.png" />
     
We see that the most frequent n-grams include many lemmas that seem to be hashtags (e.g., 'machinelearning', 'datascience', etc.). Since these have a very high frequency, considering the topics of our tweets, we have decided to remove them (hence the hashtags and mentions remover as first step above).
In addition, we see that 'data science', 'data analysis' and 'data visualization' are at the top of the two lists; which makes sense since these are the keywords the data was extracted by. So, we decided to edit the code to not keep those n-grams in. 

After removing hashtag and mention texts, and removing the aforementioned keywords, we get the following n-grams:

 1. 30 most frequent bigrams:             
<img src="https://github.com/ryabhmd/MLinPractice/blob/main/images/30_freq_bigrams_updated.png" />
               
 2. 30 most frequent trigrams:
 <img src="https://github.com/ryabhmd/MLinPractice/blob/main/images/30_freq_trigrams_updated.png" />

              
We can see that the data is still messy, as it still contains some noise; some of the things we noticed that could be further cleaned are removing words in Spanish, removing the lemma 'amp' (which is a leftover from encoding the symbol &), and using a better punctuation and stop words remover (as we still see remnants of those). Because of time constraints, we could not get to that, but we do believe these things would have given us a better result. 
From the results presented above, it seems to us that using bigrams will be more informative, as it includes more English collocations that we think might affect virality (data scientisit, big data, machine learning, etc.) as these are all current 'buzz words'.

- Choosing how to translate them as features: We chose to take each one of the 30 most common n-grams and to check whether each tweet contains the n-gram or not (binary feature).
The decision of doing it like this came from the rationale that these are tweets, and virality is affected by 'trending topics'. So, intuitevely, if the tweet contains one of the most frequent n-grams, it should reach the 'trending topics' category on twitter, and thus, more people would react to it, making the possiblity of it going viral higher.
This is why we saw no need to look into ALL n-grams, as that would be also computationally more expensive.

- Choosing 30 most frequent n-grams: We chose the 30 most frequent bigrams to be represented as binary features, because using more than 30 started to produce more and more noise text (e.g., incorrectly lemmatized, or stop words that were not actually removed, tweets in Spanish, etc.). Also, less than 30 would be too little.

# Feature Extraction
Below we explain the features we decided to extract, the rationale behind each one of them, and a brief explanation of how they were implemented.

- ``sentiment analysis``: While researching literature for what makes tweets go viral, we encountered sentiment being a big influence the most. Intuitvely as well, we think that whether a tweet is extremely positive or negative plays a role with how people react with it, and thus whether it goes viral. 
We think that mildly positive or mildly negative tweets should be the ones that associate most with virality. 
To implement this, we used nltk's SentimentIntensityAnalyzer. Which, for each sentence returns a dictionary of the following format:
                          
                          {'compound': 0.7565, 'neg': 0.092, 'neu': 0.607, 'pos': 0.301}
                          
Where 'neg', 'neu', and 'pos' give a score of how negative, neutral, and positive a string is, respectively,
and 'compund' gives a score that is set on a scale from -1 (very negative) to +1 (very positive). We use only 'compound' as a sentiment score, because it holds information on all possible sentiments in one number, which would then be easier to give to the classifier.
The SentimentIntensityAnalyzer was built and trained on social media posts; so giving it the raw tweet is the optimal format.
                          
- ``hashtags numebr``: The number of hashtags in a tweet, intuitevely, this should affect the virality, as the more hasthags there are, the more it can reach categories and topics in the 'explore' section on twitter, and thus more people react with it.

- ``mentions number``: The number of mentions in a tweet. If a tweet mentions other users, it will appear on the homepage of more users as it will get more interactions from the people mentioned. This can certainly affect virality. 

- ``urls numebr``: The number of URLs in a tweet can affect how many people interact with it and thus whether it will go more viral. 

- ``tweet length``: The length of a tweet can be a good indicator of how informative it is. In our setting, we are looking at 'scientific twitter', which probably means that people might interact more with longer, or at least medium-sized, tweets.

- ``asks for retweets or engagement``: Our intuition is that people might interact more when a user specifically asks for engagement in their tweet. So for each tweet, we checked whether it includes one or more of the following keywords: please, retweet, tweet, how, check.

- ``tells a personal story``: Our intuition is that people interact more with tweets that include some personal story or anecdote. So for each tweet, we checked whether it includes one or more of the following keywords: I, we, my, our.

- ``named entities count``: Count the number of named entities in each tweets. From looking over the tweets labeled as viral; we saw that a common pattern they had was the inclusion of some famous entities (Company names, Person names, etc.), so we decided to add the number of named entities of each tweet as a feature. We used nltk for extracting this information.

- ``contains freqeunt n-grams``: As explained in the n-grams preprocessing step above, we chose to convert the 30 most frequent bigrams to binary features, where for each tweet 'True' appears if the bigram exists in the tweet and 'False' appears otherwise.

# Unit Tests

We implemented unit tests for each step of preprocessing and feature extraction. They can all be tested automatically by running ``run_unittests.sh``.
# Dimensionality Reduction

**Curse Of Dimensionality**:
Since we have some features that include integers (tweet length, url count, mentions count, named entities count, etc.), and we have 38 features for each tweet, so we have a large feature space and need to implement dimensionality reduction. 

Since we want (to at least try) to inspect the system after reduction, to understand why the specific features were chosen, we decided to go with Feature Selection rather than Feature Projection, since the first is more interpertable. 

From all Feature Selection approaches, we found the Filter method to be the best, mainly because of not having a risk of overfitting to the overall machine learning process, and because it is computationally cheaper than the other two methods. 

Selecting number of features: we tried to do this as simple as possible, only by looking at different accuracy scores and mutual information scores, taking into account a different number of features each time. 
We monitored this simply by running the pipeline several times and checking whether there are significant changes each time we change the number of features. We started with 8 features, and saw that the results got better by getting to 10 features (accuracy scores did not change much, but MI scores were in general higher). Also, the features that were picked as the 10 best ones made sense to us; these were:

              ['tweet_charlength', 'tweet_sentiment_score', 'urls_count', 'mentions_count', 'hashtags_count', 'personal_story', 'engage_keywords', ('big', 'data'), ('we', '’'), ('open', 'data')]

With the following scores (training on KNN with K=7): 

                  training set
                    accuracy: 0.911981790113023
                    Cohen_kappa: 0.1745894733900768
                  validation set
                    accuracy: 0.9035191508062608
                     Cohen_kappa: 0.08933709391264122

When trying 11 features, we see that the accuracy does not change, neither do the MI scores; which is why we decided to stick with the 10 best features. 

# Classification

Because this is our first time implementing a machine learning pipeline, we wanted to go with a classification algorithm that we both understand well enough, and that is relatively easy for us to work with. So, we chose to work with the K Nearest Neighbor option. 
To find out what the best value of K is (i.e., the value that gives a relatively high accuracy and cohen's kappa scores), we used the implementation of the Cognitive Science Institute's grid network using the mlflow platform, as seen in the screenshot below, and in the scripts ``code/classification/classifier.sge`` and  ``code/classification/grid_search.sh``: 

<img src="https://github.com/ryabhmd/MLinPractice/blob/main/images/grid_screenshot.png" />

The only parameter we decided to change and test the the value of K in the KNN algorithm. Since we decided to go with 10 features, it made sense to test K from values 1-10 to see which one outputs the best accuracy and Cohen's kappa scores for both training and validation sets. These were the results we got: 

<img src="https://github.com/ryabhmd/MLinPractice/blob/main/images/mlflow_knn.png" />

From these results, it looks like choosing K=6 would be the best option as it has the best balance of high accuracy and Cohen's kappa score in both training and validation sets.  

Also, just to try it out, we did implement some more classification algorithms with basic parameters. This was only an experiment for us to try to see the different accuracy scores we might get, these are the results we got:

<img src="https://github.com/ryabhmd/MLinPractice/blob/main/images/mlflow_different_classifiers.png" />

We can see that the highest was the random forest classifier (at least in training set accuracy). 

# Results and Interpretation

Finally, with all the design decisions we took above (10 features, KNN classifier with K=6), we ran the classification on the test set: 
We got the following resutls: 
   
              test set
                6 nearest neighbor classifier
                  accuracy: 0.9121748390041073
                  Cohen_kappa: 0.11916745870959411


To try to make sense of this, we decided to compare it to the 'always false' baseline, since we have imbalanced data with the following label distribution: 

            False (not viral)    0.908185
            True (viral)    0.091815


This means that in an 'always false' baseline, where True Positive (TP) and False Positive (FP) classification results both equate to 0; and False Negative and True Negative classification results have a ratio of approximately 1:9, our baseline would be: 
              
              accuracy: 0.9
              Cohen_kappa: 0
              

Therefore, we can interpret that our results are barely higher than the baseline, which means that the classifier did not perform as well as we have hoped. 
We can, however, interpret this in many ways. We are aware that in some preprocessing steps we could have done a better job as we explained in the appropriate sections above, namely, in removing punctuation, filtering out unnecessary languages, a better lemmatization algorithm, etc. 
Also, as we explained above, this is our first machine learning project, which is why we chose relatively simple dimensionality reduction and classification approaches. To improve our result, we could try to use a better dimensionality reduction approach, where we implement a grid search to find the optimal number of features, as well as try different classifiers with different parameters. 
Given more time, we could have tried to improve this, but nonetheless we are satisfied with the work we did and learned a lot from it. 

# Application

The final step we did was modify the ``application.py`` file; where we added three additional functions to extract mentions, hashtags, and URLs.
These are then given to a DataFrame with the tweet that the user inserted. The DataFrame is then given to our pipleine, which outputs a prediction of how viral the inserted tweet will be. 
