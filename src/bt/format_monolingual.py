import json
import sys


def main():
    for line in sys.stdin:
        text = line[:-1]
        text = text.strip()
        if not text:
            continue
        if len(text) < 32:
            continue
        if len(text) > 512:
            continue
        print(text)


if __name__ == '__main__':
    main()
