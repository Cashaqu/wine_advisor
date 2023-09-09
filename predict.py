from summarization import get_summary
from gensim.models import Doc2Vec
from simularity import calc_sim, extract_simular_wine

model = Doc2Vec.load('./models/final_doc2vec.model')

print("Welcome to Wine-Advisor!")
phrase = input("Enter a description here and I will try to find a suitable wine for you:\n")

print(f"Let's shorten your description to this form:\n {get_summary(phrase)}")
phrase_calculate = calc_sim(phrase)
extract_simular_wine(phrase_calculate)