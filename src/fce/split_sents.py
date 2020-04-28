import sys

from spacy.lang.en import English


def main():
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    for line in sys.stdin:
        src, tgt = line[:-1].split('\t')
        src_sents = list(nlp(src).sents)
        tgt_sents = list(nlp(tgt).sents)
        if len(src_sents) != len(tgt_sents):
            print('***')
            print(line)
            continue
        for src_sent, tgt_sent in zip(src_sents, tgt_sents):
            print(f'{src_sent}\t{tgt_sent}')


if __name__ == '__main__':
    main()
