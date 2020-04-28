import aspell
import re
import sys
from spacy.lang.en import English

WHITELIST = {"'m",  "'s", "'ll", "n't", "'d", "'ve"}


def tokenize(text, nlp):
    results = []
    prev_token_end = None
    for token in nlp(text):
        if token.idx == 0:
            space_before = False
        else:
            space_before = prev_token_end != token.idx

        results.append((str(token), space_before))
        prev_token_end = token.idx + len(token)

    return results


def correct(tokens, speller):
    results = []
    for token, space_before in tokens:
        if speller.check(token):
            results.append((token, space_before))
        else:
            suggestions = speller.suggest(token)
            if suggestions:
                token_corrected = suggestions[0]
            else:
                token_corrected = token

            if token_corrected == 'W' or token.lower() in WHITELIST or re.search(r'^[0-9]', token):
                # Aspell tends to correct any irregular special symbols (e.g., '--' and '...') as W
                token_corrected = token
            results.append((token_corrected, space_before))
    return results


def untokenize(tokens):
    results = []
    for token, space_before in tokens:
        if space_before:
            results.append(' ')
        results.append(token)

    return ''.join(results)


def main():
    nlp = English()
    speller = aspell.Speller('lang', 'en')

    for line in sys.stdin:
        text = line[:-1]
        tokens = tokenize(text, nlp=nlp)
        tokens = correct(tokens, speller=speller)
        text = untokenize(tokens)
        print(text)


if __name__ == '__main__':
    main()
