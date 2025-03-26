import random
import nltk
from nltk.corpus import words


nltk.download('words')

all_words = set(words.words())

common_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 'it', 'for', 'not', 'on', 'with', 'he', 'she', 'we', 'you', 'they', 'at', 'this', 'but', 'had', 'by', 'from', 'or', 'as', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'which', 'who', 'how', 'me', 'them', 'so', 'up', 'out', 'if', 'about', 'into', 'just', 'now', 'your', 'has', 'more', 'can', 'than', 'its', 'also'}

filtered_words = [word for word in all_words if word.lower() not in common_words]

selected_words = random.sample(filtered_words, 1000)

with open('tokens.txt', 'w') as file:
    for word in selected_words:
        file.write(f"{word}\n")

print("1000 unique words saved to tokens.txt")