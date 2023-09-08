from gensim.models import Doc2Vec
from datetime import datetime
from utils import ExecutionTime


def build_model(max_epochs, vec_size, alpha, tagged):
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

    model.save('./models/' + (datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')) + '_doc2vec.model')
    print(f'Training completed: {full_time:.2f} sec')
    print("Model saved to ./models")
    return model