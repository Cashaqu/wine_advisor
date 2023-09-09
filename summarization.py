from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

from utils import ExecutionTime, write_list, read_list
from tqdm import tqdm

summarizer = TextRankSummarizer()


def get_summary(text: str):
    sum_sent = ''
    parser = PlaintextParser.from_string(text, Tokenizer('english'))
    summary = summarizer(parser.document, 2)
    for sentence in summary:
        sum_sent += (str(sentence)) + ' '
    return sum_sent


def extracting_summary(path: str, is_saved=True):
    description_list = read_list(path)
    t = ExecutionTime()
    t.start()
    sum_description = []
    print('Summarization started...')
    for description in tqdm(description_list):
        sum_description.append(get_summary(description))
    t.end()
    print(f'Summarization completed in {t.get_exec_time():.2f} sec')
    if is_saved:
        path = './data/summarization_text'
        write_list(path, sum_description)
        print(f'Summary saved to {path}')
    return sum_description
