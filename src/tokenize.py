import re
import sys

ALL_CHARS = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?'-")


def tokenize_characters(text):
    text = text.strip()
    text = ''.join(ch if ch in ALL_CHARS else '#' for ch in text)
    text = re.sub(' +', ' ', text).strip()
    tokens = [ch if ch != ' ' else '▁' for ch in text]
    return tokens


if __name__ == '__main__':
    for line in sys.stdin:
        text = line[:-1]
        tokens = tokenize_characters(text)[:1023]
        print(' '.join(tokens))
