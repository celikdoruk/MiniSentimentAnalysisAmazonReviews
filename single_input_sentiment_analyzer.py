print('The application is being loaded...')

import warnings
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

# moadel initialize
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
sia = SentimentIntensityAnalyzer()
wnl = WordNetLemmatizer()
warnings.filterwarnings('ignore')
stip_words_list = stopwords.words('english')


def nltk_analyze(text):
    text = nltk.word_tokenize(text)
    cleaned_text = [wnl.lemmatize(word) for word in text if word.isalpha() and word not in stip_words_list]
    final_text = ' '.join(cleaned_text)
    return sia.polarity_scores(final_text)


def roberta_analyze(text):
    try:
        text = tokenizer(text, return_tensors='pt')
        output = model(**text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
    except:
        print('There was an error within your text.')
    return scores


### Beginnning of the application

while True:
    print('\nWelcome to Sentiment Analyzer!\n(To exit or go back in the program, press 9)\n\nFor VADER Model, Press 1\nFor RoBERTa Model, Press 2\n')
    user_input = int(input('Your choice: '))

    if user_input == 9:
        print('Closing the application...')
        break

    if user_input == 1:
        print('\nVADER Model in use.')
        
        while True:
            text = input('\nGo ahead and write your sentence: ')

            if text == '9':
                break
            
            nltk_values = nltk_analyze(text)
            neg = nltk_values['neg']
            neu = nltk_values['neu']
            pos = nltk_values['pos']

            sentiment = sorted([neg, neu, pos])[::-1]

            if sentiment[0] == neg:
                print('This seems to be a negative sentence.')
            elif sentiment[0] == neu:
                print('This seems to be a neutral sentence.')
            elif sentiment[0] == pos:
                print('This seems to be a positive sentence.')
            else:
                print('There was an error within your text.')


    elif user_input == 2:
        print('\nRoBERTa Model in use.')

        while True:
            text = input('\nGo ahead and write your sentence: ')

            if text == '9':
                break
            
            roberta_values = roberta_analyze(text)
            neg = roberta_values[0]
            neu = roberta_values[1]
            pos = roberta_values[2]

            sentiment = sorted([neg, neu, pos])[::-1]

            if sentiment[0] == neg:
                print('This seems to be a negative sentence.')
            elif sentiment[0] == neu:
                print('This seems to be a neutral sentence.')
            elif sentiment[0] == pos:
                print('This seems to be a positive sentence.')
            else:
                print('There was an error within your text.')
    else:
        print('Invalid number.')
    