from gensim.models.doc2vec import TaggedDocument
from utils import ExecutionTime


def tagging(cleaned_text: list):
    print('Tagging started...')
    t = ExecutionTime()
    t.start()
    idx = [str(i) for i in range(len(cleaned_text))]
    tagged_text = []
    for i in range(len(cleaned_text)):
        tagged_text.append(TaggedDocument(cleaned_text[i], [idx[i]]))
    t.end()
    print(f'Tagging completed in {t.get_exec_time():.2f} sec')
    return tagged_text
