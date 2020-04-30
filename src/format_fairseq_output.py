import re
import sys


def main():
    prev_line_no = None
    for line in sys.stdin:
        line = line[:-1]
        match = re.match(r'^H-(\d+)', line)
        if not match:
            continue
        tokens = line.split('\t')[2].split(' ')
        text = ''.join(tokens)
        line_no = int(match.group(1))
        assert not prev_line_no or line_no == prev_line_no + 1
        prev_line_no = line_no
        text = text.replace('▁', ' ')
        print(text)


if __name__ == '__main__':
    main()
