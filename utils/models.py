import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess


def load_assets():
    model = load_model('assets/best_lstm_model.h5')
    with open('assets/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    

    return model, tokenizer

