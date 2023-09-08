from gensim.parsing.preprocessing import remove_stopwords, remove_short_tokens
from gensim.parsing.preprocessing import strip_punctuation

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from utils import ExecutionTime


def to_lower(text: list):
    return [symbol.lower() for symbol in text]


def removing_stopword(text: list):
    return [remove_stopwords(word) for word in text]


def removing_punctuation(text):
    return [strip_punctuation(symbol) for symbol in text]


def tokenization(text: list):
    return [word_tokenize(word) for word in text]


def removing_short_tokens(text: list):
    return [remove_short_tokens(tokens) for tokens in text]


def creating_lemmas(text: list):
    wnl = WordNetLemmatizer()
    return [[wnl.lemmatize(word) for word in sentence] for sentence in text]


def cleaning_text(text: list):
    print('Cleaning text started...')
    t = ExecutionTime()
    t.start()
    text = to_lower(text)
    text = removing_stopword(text)
    text = removing_punctuation(text)
    text = tokenization(text)
    text = creating_lemmas(text)
    text = removing_short_tokens(text)
    t.end()
    print(f'Text cleaning completed in {t.get_exec_time():.2f} sec')
    return text
