# Introduction to Natural Language Processing

### Sentiment Analysis on Amazon Reviews

This project showcases basic sentiment analysis using two models: VADER (from NLTK) and RoBERTa (from Hugging Face Transformers), applied to Amazon product reviews. It includes a terminal-based sentiment analyzer app and a Jupyter Notebook that demonstrates data preprocessing and sentiment scoring.

🔍 What This Project Does

* Preprocesses review text (tokenization, lemmatization, stopword removal)
* Applies VADER for rule-based sentiment scoring
* Applies RoBERTa for deep learning-based sentiment scoring
* Compares model results and explores data patterns

📁 Project Structure

* sentiment_analysis_amazon_final.ipynb — NLP preprocessing and sentiment scoring on a dataset
* single_input_sentiment_analyzer.py — A terminal-based Python sentiment analysis tool using user input (includes both NLTK's VADER and Hugging Face's RoBERTa)
* amazon_reviews.csv — Dataset used

📊 Dependencies

* PyTorch, Tenserflow 2.0 or later.
* NLTK
* Hugging Face Transformers
