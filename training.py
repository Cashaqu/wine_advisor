from gensim.models import Doc2Vec
from datetime import datetime
from utils import ExecutionTime


def build_model(max_epochs, vec_size, alpha, tagged, is_saved=True):
    """
    Initializes doc2vec model and trains it.
        Args:
            max_epochs: max epochs for training
            vec_size: how much size vector for representation sentence
            is_saved: if True, list of summary will be saved
            alpha: learning rate in begin
            tagged:
            is_saved: is_saved: if True, model will be saved
        Return:
            Model doc2vec.
    """

    model = Doc2Vec(vector_size=vec_size, alpha=alpha, min_alpha=0.00025, min_count=1, dm=1)
    model.build_vocab(tagged)
    print('Training started...')
    t = ExecutionTime()
    full_time = 0

    for epoch in range(max_epochs):
        t.start()
        model.train(tagged,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha
        t.end()
        full_time += t.get_exec_time()
        print(f'Epoch: {epoch}; Execution time: {t.get_exec_time():.2f} sec')

    print(f'Training completed: {full_time:.2f} sec')
    if is_saved:
        model.save('./models/' + (datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')) + '_doc2vec.model')
        print("Model saved to ./models")
    return model
