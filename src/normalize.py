import sys

from src.tokenize import tokenize_characters


def main():
    for line in sys.stdin:
        text = line[:-1]
        tokens = tokenize_characters(text)[:1023]
        text = ''.join(tokens)
        text = text.replace('▁', ' ')
        print(text)


if __name__ == '__main__':
    main()
