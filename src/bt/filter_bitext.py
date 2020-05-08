import json
import sys
import editdistance


def main():
    for line in sys.stdin:
        src_text, tgt_text = line[:-1].split('\t')
        edit_dist = editdistance.eval(src_text, tgt_text)
        edit_ratio = edit_dist / max(len(src_text), len(tgt_text))
        if edit_dist > 32 or edit_ratio > .1:
            continue
        print(line[:-1])


if __name__ == '__main__':
    main()
