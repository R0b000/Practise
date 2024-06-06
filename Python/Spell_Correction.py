import re
import sys
from collections import Counter, defaultdict
import nltk
from nltk.corpus import words as nltk_words
from nltk.util import ngrams

# NLTK words
nltk_words_set = set(nltk_words.words())

# Load your big.txt file or any other large text corpus
with open('big.txt', 'r') as file:
    text = file.read()

# Preprocess text
def words(text):
    return re.findall(r'\w+', text.lower())

# Tokenize the text and build vocabulary
def build_vocab(text):
    return Counter(words(text))

# Add NLTK words to the corpus for a richer vocabulary
vocab = build_vocab(text)
vocab.update(nltk_words_set)

# Create a frequency dictionary of words
WORD_FREQ = defaultdict(int)
for word, freq in vocab.items():
    WORD_FREQ[word] += freq

# Calculate total number of words
N = sum(WORD_FREQ.values())

# Function to get probability of a word
def P(word):
    "Probability of `word`."
    return WORD_FREQ[word] / N

# Generate possible spelling corrections for word
def candidates(word):
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

# Subset of words that appear in the dictionary of WORDS
def known(words):
    return set(w for w in words if w in WORD_FREQ)

# All edits that are one edit away from word
def edits1(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

# All edits that are two edits away from word
def edits2(word):
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

# Function to get n-grams from text
def get_ngrams(text, n):
    tokens = words(text)
    n_grams = ngrams(tokens, n)
    return [' '.join(gram) for gram in n_grams]

# Correct text
def correct_text(text):
    corrected_text = []
    words = re.findall(r'\w+|\S', text)  # Include punctuation as separate tokens

    for word in words:
        if word.isalpha():
            if word[0].isupper():
                corrected_word = correction(word.lower()).capitalize()
            else:
                corrected_word = correction(word)
        else:
            corrected_word = word
        corrected_text.append(corrected_word)

    return " ".join(corrected_text)

# Most probable spelling correction for word
def correction(word):
    return max(candidates(word), key=P)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python spell_corrector.py <input_text>")
        sys.exit(1)

    input_text = sys.argv[1]
    corrected_text = correct_text(input_text)

    print(f"Original: {input_text}")
    print(f"Corrected: {corrected_text}")
