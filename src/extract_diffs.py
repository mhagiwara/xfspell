import difflib
import sys


def get_edits(text1, text2):
    edits = []

    s = difflib.SequenceMatcher(None, text1, text2, autojunk=False)
    for ops in s.get_grouped_opcodes(1):
        piece1, piece2 = '', ''
        first_i1 = None
        for op in ops:
            tag, i1, i2, j1, j2 = op
            piece1 += text1[i1:i2]
            piece2 += text2[j1:j2]
            if first_i1 is None:
                first_i1 = i1
        edits.append((first_i1, piece1, piece2))

    return edits


def main():
    for line in sys.stdin:
        line = line[:-1]
        src, tgt = line.split('\t')
        edits = get_edits(src, tgt)
        edits = [f'{first_i1}:{piece1}:{piece2}' for first_i1, piece1, piece2 in edits]
        print(' '.join(edit.replace(' ', '_') for edit in edits))


if __name__ == '__main__':
    main()
