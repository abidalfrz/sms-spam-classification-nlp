from tensorflow.keras.models import load_model
import pickle

def load_assets():
    model = load_model('assets/best_lstm_model.h5')
    with open('assets/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    

    return model, tokenizer

