import pandas as pd
import pandas as pd
import pymorphy2
from rutermextract import TermExtractor
import spacy_udpipe
import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
nltk.download("punkt")
nltk.download('stopwords')
russian_stopwords = stopwords.words("russian")
morph = pymorphy2.MorphAnalyzer()

from src.constants import DATASET

def tag_np_russian(text):
    words = text.split()
    tagged_text = []
    i = 0
    while i < len(words):
        word = words[i]
        parsed_word = morph.parse(word)[0]

        if 'NOUN' in parsed_word.tag:
            np = [word]
            i += 1
            while i < len(words):
                next_word = words[i]
                parsed_next_word = morph.parse(next_word)[0]

                if 'NOUN' in parsed_next_word.tag:
                    np.append(next_word)
                    i += 1
                else:
                    break
            if len(np) > 1:
              tagged_text.append((" ".join(np), "NP"))
        else:
            i += 1
    return tagged_text