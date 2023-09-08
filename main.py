from extract_description import extract_description
from preprocessing import cleaning_text
from tagging import tagging
from training import build_model

if __name__ == '__main__':
    corpus = extract_description('./data/wine_reviews.csv', 'description')
    clean_text = cleaning_text(corpus)
    tagged_text = tagging(clean_text)
    model = build_model(max_epochs=10, vec_size=50, alpha=0.025, tagged=tagged_text)
