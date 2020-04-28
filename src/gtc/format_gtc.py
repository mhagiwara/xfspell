import json
import sys
from src.tokenize import tokenize_characters


def main():
    for line in sys.stdin:
        data = json.loads(line)
        for edit in data['edits']:
            lang = edit['src']['lang']
            if lang != 'eng':
                continue
            if not edit['is_typo']:
                continue
            src, tgt = edit['src']['text'], edit['tgt']['text']
            src = src.strip().replace('\t', ' ')
            tgt = tgt.strip().replace('\t', ' ')
            print(f'{src}\t{tgt}')


if __name__ == '__main__':
    main()
