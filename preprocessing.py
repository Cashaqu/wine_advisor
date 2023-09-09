from gensim.parsing.preprocessing import remove_stopwords, remove_short_tokens
from gensim.parsing.preprocessing import strip_punctuation

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from utils import ExecutionTime


def to_lower(text: list) -> list[str]:
    """
    Сonverts all characters in a list of string to lowercase.
        Args:
            text: Input text
        Return:
            A list of string in lowercase. For example:
                ['This wine super cool. Drink it every day.'] ->
                ['this wine super cool. drink it every day.']
    """
    return [symbol.lower() for symbol in text]


def removing_stopword(text: list) -> list[str]:
    """
    Removes all stopwords in sentence.
        Args:
            text: Input text
        Return:
            A list of string without stopwords. For example:
                ['This wine super cool. Drink it every day.'] ->
                ['This wine super cool. Drink day.']
    """
    return [remove_stopwords(word) for word in text]


def removing_punctuation(text) -> list[str]:
    """
    Removes punctuation in sentence.
        Args:
            text: Input text
        Return:
            A list of string without punctuation. For example:
                ['Know what? This wine is great, fantastic! Drink it every day.'] ->
                ['Know what  This wine is great  fantastic  Drink it every day ']
    """
    return [strip_punctuation(symbol) for symbol in text]


def tokenization(text: list) -> list[list[str]]:
    """
    Tokenizes every word in sentence to token.
        Args:
            text: Input text
        Return:
            A list of lists with words without stopwords. For example:
            ['This wine super cool.', 'Drink it every day.'] ->
            [['This', 'wine', 'super', 'cool', '.'], ['Drink', 'it', 'every', 'day', '.']]
    """
    return [word_tokenize(word) for word in text]


def removing_short_tokens(text: list) -> list[list[str]]:
    """
    Removes tokens whose length is less than a specified value (default <3).
        Args:
            text: Input text
        Return:
            A list of lists strings without short tokens. For example:
    [['It', "'s", 'delicious', 'now', ',', 'with', 'its', 'light']] ->
    [['delicious', 'now', 'with', 'its', 'light']]
    """
    return [remove_short_tokens(tokens) for tokens in text]


def creating_lemmas(text: list) -> list[list[str]]:
    """
    Removes tokens whose length is less than a specified value (default <3).
        Args:
            text: Input text
        Return:
            A list of lists strings with lemmas. For example:
    [['its', 'light', 'tannins', 'and', 'bright', 'raspberry', 'aftertaste']] ->
    [['it', 'light', 'tannin', 'and', 'bright', 'raspberry', 'aftertaste']]
    """
    wnl = WordNetLemmatizer()
    return [[wnl.lemmatize(word) for word in sentence] for sentence in text]


def cleaning_text(text: list) -> list[list[str]]:
    """Performs all operations above.
        Args:
            text: Input text
        Return:
            A list of lists 'cleaned' words. For example:
    ["It's delicious now, with its light tannins and bright raspberry aftertaste"] ->
    [['delicious', 'now', 'light', 'tannin', 'bright', 'raspberry', 'aftertaste']]
    """
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
