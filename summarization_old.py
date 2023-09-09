from utils import write_list
from tqdm import tqdm
from utils import ExecutionTime

from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
import spacy.cli

if not spacy.util.is_package("en_core_web_sm"):
    spacy.cli.download("en_core_web_sm")
nlp = spacy.load('en_core_web_sm')




class Summarization:
    def __init__(self, doc: str):
        self.doc = nlp(doc)

    def filtering(self):
        keywords = []
        stopwords = list(STOP_WORDS)
        pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
        for token in self.doc:
            if token.text in stopwords or token.text in punctuation:
                continue
            if token.pos_ in pos_tag:
                keywords.append(token.text)
        return keywords

    def normalizing(self):
        freq_words = Counter(self.filtering())
        max_freq = freq_words.most_common(1)[0][1]

        for word in freq_words.keys():
            freq_words[word] /= max_freq
        return freq_words

    def get_weights(self):
        sent_strength = {}
        freq_words = self.normalizing()
        for sent in self.doc.sents:
            for word in sent:
                if word.text in freq_words.keys():
                    if sent in sent_strength.keys():
                        sent_strength[sent] += freq_words[word.text]
                    else:
                        sent_strength[sent] = freq_words[word.text]
        return sent_strength

    def get_summary(self):
        x = self.get_weights()
        summarized_sentences = nlargest(3, x, key=x.get)
        final_sentences = [w.text for w in summarized_sentences]
        summary = ' '.join(final_sentences)
        return summary


def extracting_summary(description_list: list, is_saved=True):
    t = ExecutionTime()
    t.start()
    sum_description = []
    print('Summarization started...')
    for description in tqdm(description_list):
        description = Summarization(description)
        sum_description.append(description.get_summary())
    t.end()
    print(f'Summarization completed in {t.get_exec_time():.2f} sec')
    if is_saved:
        path = './data/summarization_text'
        write_list(path, sum_description)
        print(f'Summary saved to {path}')
    return sum_description

