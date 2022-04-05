"""
Input: Iliad from Homer
Output: sentiment analysis of the Iliad
"""

import re
import requests
import string
import json

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

sia = SentimentIntensityAnalyzer()

# Download nltk data (if required)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')


def count_words(iliad: str) -> int:
    return len(iliad.split(' '))


def download_Iliad() -> str:
    # Download The Iliad from Homer (from gutenberg.org)
    url = 'https://www.gutenberg.org/cache/epub/16452/pg16452.txt'
    return requests.get(url).text


def get_Iliad() -> str:
    iliad = download_Iliad()

    index_end_header = iliad.find("ENGLISH BLANK VERSE.")
    index_start_footer = iliad.find("FOOTNOTES")

    return iliad[index_end_header: index_start_footer]


def remove_book_headers(iliad: str) -> str:
    """ 
    Remove the book headers (as they are not part of the original poem).

    For instance:

        BOOK I.


        ARGUMENT OF THE FIRST BOOK.

        The book opens with an account of a pestilence that prevailed in the
        Grecian camp, and the cause of it is assigned. A council is called, in
        which fierce altercation takes place between Agamemnon and Achilles.
        The latter solemnly renounces the field. Agamemnon, by his heralds,
        demands Brisëis, and Achilles resigns her. He makes his complaint to
        Thetis, who undertakes to plead his cause with Jupiter. She pleads it,
        and prevails. The book concludes with an account of what passed in
        Heaven on that occasion.


        [The reader will please observe, that by Achaians, Argives, Danaï, are
        signified Grecians. Homer himself having found these various
        appellatives both graceful and convenient, it seemed unreasonable that
        a Translator of him should be denied the same advantage.—Tr.]
    """

    poem = []

    # keep track of the "book markers" in a set
    # (example of book marker: `BOOK I.`)
    book_markers = set()

    # define a variable `is_poem` that will _alternatively_
    # have the boolean value True and False
    is_poem = True

    for sentence in iliad.split('\r\n'):

        # if there is a book marker (i.e., starting with "BOOK ")...
        if 'BOOK ' in sentence:
            # ... if this marker is NOT already in the set of markers,
            # then it means that this is a book header that is not part
            # of the poem (i.e., this is the first time that this book
            # marker is seen). Therefore, set `is_poem` to False and add
            # this book marker to the set of book markers
            if sentence not in book_markers:
                is_poem = False
                book_markers.add(sentence)
            # ... otherwise, this marker has already been seen, meaning that
            # the book header has already been ignored. Therefore, `is_poem`
            # can be set to True
            else:
                is_poem = True

        # if and only if this is a poem, add the sentence to the list of poetic
        # sentences
        if is_poem:
            poem.append(sentence)

    # transform the list of sentences into one string (programmatic idiom)
    return ' '.join(poem)


def clean_poem(iliad: str) -> str:
    """
    Clean the poem by removing digits and translator's comments
    """

    # 1. remove the book headers (they are indeed not part of the original poem)
    iliad = remove_book_headers(iliad)

    # 1. using a regex, replace the newlines with a space
    # `\r\n` => new line
    iliad = re.sub('\r\n', ' ', iliad)
    # 2. remove all digits using the maketrans+translate idiom
    iliad = iliad.translate(str.maketrans('', '', string.digits))
    # 3. remove the comments from the translator:
    # 3a. first, remove everything between square brackets
    #     (see: https://stackoverflow.com/a/14599280)
    iliad = re.sub("[\[].*?[\]]", "", iliad)
    # 3b. then, remove the `-Tr`
    iliad = iliad.replace('—Tr.', '')
    # 4. replace consecutive spaces with only one
    iliad = re.sub('\s+', ' ', iliad)

    return iliad


def modernize_sentences(iliad: str) -> str:
    """
    Just `’d` with `ed`
    """
    return iliad.replace('’d', 'ed')


def remove_stopwords(iliad: str) -> str:
    """
    Remove the stopwords as they are useless to the sentiment
    analysis
    """
    stop_words = set(stopwords.words('english'))
    iliad_without_stopwords = [w for w in iliad.split(
        ' ') if not w.lower() in stop_words]

    return ' '.join(iliad_without_stopwords)


def sentiment_analysis(iliad: str) -> None:
    """
    Compute the sentiment analysis
    """

    print(f"Analyzing {count_words(iliad)} words...")

    scores = sia.polarity_scores(iliad)

    print("Results of the sentiment analysis:")

    # just pretty print the score object (interpreted as a JSON)
    print(json.dumps(scores, indent=4))


iliad = get_Iliad()

print(f"The download text contains {count_words(iliad)} words")

iliad = clean_poem(iliad)
iliad = modernize_sentences(iliad)

print(f"The Iliad contains {count_words(iliad)} words")

iliad = remove_stopwords(iliad)

print(f"The Iliad without stopwords contains {count_words(iliad)} words")

sentiment_analysis(iliad)
