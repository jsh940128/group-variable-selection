# Data Preprocessing
import pandas as pd
import numpy as np

# Type Hinting
from typing import Union

# Lemmatization
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class WordFrequencyGenerator(object):
    def __init__(
        self, 
        data_frame: pd.DataFrame,
        review_col: str,
        rating_col: str,
        aggregate_col: Union[str, None],
        word_corpus_file = None,
        n_jobs = 1
    ):
        self.data_frame = data_frame
        self.review_col = review_col
        self.rating_col = rating_col
        self.aggregate_col = aggregate_col
        self.word_corpus_avail = False
        self.n_jobs = n_jobs
        
        if word_corpus_file != None:
            self.word_corpus = self._load_corpus_file(word_corpus_file)
            self.word_corpus_avail = True

    def _load_corpus_file(self, word_corpus_file):
        word_corpus = pd.read_excel(word_corpus_file)
        word_corpus.columns = ['valid_corpus', 'change']
        word_corpus = word_corpus.dropna(subset = ['valid_corpus'])
        word_corpus['change'] = word_corpus['change'].fillna(word_corpus['valid_corpus'])

        return dict(zip(word_corpus['valid_corpus'], word_corpus['change']))
    
    def lemmatize(self, data_frame):
        review_col = self.review_col
        
        stop_words = set(stopwords.words('english'))
        stop_words = stop_words.union({'.','\'', '\'\'', ',', ':', '(', ')', '!', '-', '[', ']', 'n\'t', '\'s', '#', '$', '%', '&', ''', '''', ''d', '\'ll', '\'m', '\'re', '\'ve', '*',
               '+', '++', '--', '..', '...', '....', '.....', '/', '//',
               '//www.youtube.com/watch', '/b', '/bgcolor', '/color', '/i', '/size',
               '/thing', '/u', '/url','zu', '{', '|',
               '}', '~', 'à', 'è', 'é', 'és', 'в', 'и', 'на', 'не', '–', '—', '‘', '’',
                "'d", '2nd', '2p', ':nostar', ':star', ';', '=', '>', '?', '``',
               '“', '”', '•'})
        lemmatizer = WordNetLemmatizer()
        verb_lemma = lambda x: lemmatizer.lemmatize(x, pos = 'v')
        noun_lemma = lambda x: lemmatizer.lemmatize(x, pos = 'n')
        
        if self.word_corpus_avail == True:
            data_frame['corpus'] = data_frame[review_col].apply(
                lambda x: [
                    self.word_corpus[noun_lemma(verb_lemma(word)).lower()] 
                    for word in word_tokenize(x) if noun_lemma(verb_lemma(word)).lower() in self.word_corpus
                ]
            )
        else:
            data_frame['corpus'] = data_frame[review_col].apply(
                lambda x: [
                    noun_lemma(verb_lemma(word)).lower() 
                    for word in word_tokenize(x) if noun_lemma(verb_lemma(word)).lower() not in stop_words
                ]
            )
            
        return data_frame[data_frame[review_col].apply(len) > 0]
    
    def generate(self, word_count = None, top_n = None) -> pd.DataFrame:
        """Generate Word Frequency DataFrame. Each row has an original index from the aggregator.
        
        Parameter
        ========
        word_count: Threshold to select words, based on frequency
        top_n: Select Top N words that occurred in the document
        """
        if (word_count != None) and (top_n != None):
            raise ValueError("Invalid argument! Please input only one of `word count` or `top n`")

        lemmatized_df = self.lemmatize(self.data_frame)
        lemmatized_df['intercept'] = 1
        
        word_df = lemmatized_df.explode('corpus')
        word_df['intercept'] = 1
        word_tabulation = word_df['corpus'].value_counts()
        
        if word_count != None:
            select_word = word_tabulation[word_tabulation > word_count].index
        elif top_n != None:
            select_word = word_tabulation[:top_n].index
        else:
            select_word = word_tabulation.index
        print(f"{len(select_word)} of words are selected")
        
        frequency_matrix = word_df[word_df['corpus'].isin(select_word)].pivot_table(
            index = self.aggregate_col,
            columns = 'corpus',
            values = 'intercept',
            aggfunc = 'count'
        ).fillna(0)
        
        weight_vector = lemmatized_df[
            lemmatized_df[self.aggregate_col].isin(frequency_matrix.index)
        ].groupby(self.aggregate_col)\
         .agg({
            'intercept': 'count',
             self.rating_col: 'mean'
        }).rename(
             columns = {
                 'intercept': 'count',
                 self.rating_col: 'mean_rating'
             }
        )
        self.frequency_matrix = frequency_matrix
        self.weight_vector = weight_vector
        
        return frequency_matrix, weight_vector

    def find_appropriate_document(self, *arg):
        condition = (self.frequency_matrix[[*arg]] >= 1).sum(axis = 1)
        index = self.frequency_matrix.loc[condition == len(arg), :].index
        return self.data_frame.loc[self.data_frame[self.aggregate_col].isin(index), [self.aggregate_col, self.rating_col, self.review_col]]

    def export_word_corpus(self, file_name):
        """Export to excel file"""
        pd.DataFrame(self.frequency_matrix.columns).to_excel(file_name, index = False)










        