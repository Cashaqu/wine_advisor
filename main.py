from preprocess_df import preprocess_df
from extract_description import extract_description
from preprocessing import cleaning_text
from tagging import tagging
from training import build_model
from utils import read_list
from summarization import extracting_summary
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Get some hyperparameters.")

    # Get an arg for num_epochs
    parser.add_argument("--num_epochs",
                        default=5,
                        type=int,
                        help="the number of epochs to train for")

    # Get an arg for vec_size
    parser.add_argument("--vec_size",
                        default=50,
                        type=int,
                        help="vector size for doc2vec")

    # Get an arg for alpha
    parser.add_argument("--alpha",
                        default=0.025,
                        type=float,
                        help="learning rate to use for model")


    args = parser.parse_args()

    NUM_EPOCHS = args.num_epochs
    VEC_SIZE = args.vec_size
    ALPHA = args.alpha


    # dataset preparation
    preprocess_df('./data/wine_reviews.csv', './data/cleaned_wine_reviews.csv')

    # extracting description to file
    corpus = extract_description('./data/cleaned_wine_reviews.csv', 'description')

    # summarazing text and saving to file
    summary_text = extracting_summary('./data/description')

    # cleaning text
    clean_text = cleaning_text(read_list('./data/summarization_text'))

    # tagging text
    tagged_text = tagging(clean_text)

    # build and train model
    print(f'[INFO] Number of epochs: {NUM_EPOCHS}')
    print(f'[INFO] Vector size: {VEC_SIZE}')
    print(f'[INFO] Alpha: {ALPHA}')

    model = build_model(max_epochs=NUM_EPOCHS,
                        vec_size=VEC_SIZE,
                        alpha=ALPHA, tagged=tagged_text)
