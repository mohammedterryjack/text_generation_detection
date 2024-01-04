from src.utils import bag_of_words, count_punctuation, normalise, find_quotes

def novelty(answer:str, source_text:str) -> float:
    source_words = bag_of_words(text=source_text)
    answer_words = bag_of_words(text=answer)
    answer_novel_words = answer_words - source_words
    n_novel_words = len(answer_novel_words)
    n_words = len(answer_words)
    return n_novel_words/n_words

def punctuation_density(text:str) -> float:
    n_punctuation = count_punctuation(text=text)
    n_characters = len(text)
    return n_punctuation/n_characters

def quote_density(answer:str, source_text:str) -> float:
    quotes = find_quotes(
        answer=answer,
        source_text=source_text
    )
    n_characters_quotes = sum(map(len,quotes))
    n_characters = len(source_text)
    return n_characters_quotes/n_characters

def non_stopword_density(text:str) -> float:
    non_stopwords = normalise(text).split()
    words = text.split()
    n_non_stopwords = len(non_stopwords)
    n_words = len(words)
    return n_non_stopwords/n_words

def get_features(answer:str, source_text:str) -> dict[str,float]:
    return dict(
        information_density=non_stopword_density(text=answer),
        quote_density_score = quote_density(
            answer=answer,
            source_text=source_text
        ),
        punctuation_density_score = punctuation_density(text=answer),
        novelty_score = novelty(answer=answer, source_text=source_text)
    )

#TODO: more efficient version if you calculate these features at same time and re-use certain functions like normalise