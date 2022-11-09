#Native lenguage ToolKit
import nltk
import numpy as np
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

#Define stemmer
stemmer = PorterStemmer()

#Split string in word vectpr
def tokenize(sentence):
    return nltk.word_tokenize(sentence)


#input: string -> root of word
def stem(word):
    return stemmer.stem(word.lower())

#Words codification
def bag_of_words(tokenized_sentence, all_words):
    #stem each word
    sentence_word = [stem(w) for w in tokenized_sentence]

    #zero initial vector
    bag = np.zeros(len(all_words), dtype=np.float32)
    
    #enumerate: move arrays side by side
    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    
    return bag




