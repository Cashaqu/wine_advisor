from preprocessing import cleaning_text
from gensim.models import Doc2Vec
import pandas as pd
from prettytable import PrettyTable
def calc_sim(query: str) -> list[int]:
    """
    Function for calculating simular wines by query.
        Args:
            query: query that comes from console input
        Return:
            List with 5 integer element (indexes of most simular wines in 'wine_reviews.csv'). For example:
            [4, 8, 15, 16, 23]
    """

    model = Doc2Vec.load('./models/final_doc2vec.model')
    clean_query = cleaning_text([query])
    vector_query = model.infer_vector(clean_query[0])
    similar_sentences = model.dv.most_similar(positive=[vector_query])
    return [int(similar_sentences[i][0]) for i in range(0,5)]

def extract_simular_wine(sim_idx: list):
    """
        Function to display output in a beautiful way
            Args:
                sim_idx: List with 5 integer element (indexes of most simular wines in 'wine_reviews.csv')
            Return:
                List with 5 integer element (indexes of most simular wines in 'wine_reviews.csv'). For example:
                                            +-------+---------+--------+-------------+
                                            | title | variety | points | description |
                                            +-------+---------+--------+-------------+
                                            |  ...  |   ...   |  ...   |     ...     |
                                            |  ...  |   ...   |  ...   |     ...     |
                                            |  ...  |   ...   |  ...   |     ...     |
                                            |  ...  |   ...   |  ...   |     ...     |
                                            +-------+---------+--------+-------------+
    """
    df = pd.read_csv('./data/wine_reviews.csv')
    df_idx = df.iloc[sim_idx]
    x = PrettyTable()
    x.field_names = ["title", "variety", "points", "description"]
    for idx in df_idx.index:
        x.add_row([df_idx['title'][idx],
                   df_idx['variety'][idx],
                   df_idx['points'][idx],
                   df_idx['description'][idx]])
    print(x)
