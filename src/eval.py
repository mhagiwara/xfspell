import sys


def main():
    pred_path = sys.argv[1]
    gold_path = sys.argv[2]

    pred_all_edits = set()
    with open(pred_path) as f:
        for i, line in enumerate(f):
            line = line[:-1]
            if not line:
                continue
            for edit in line.split(' '):
                pred_all_edits.add((i, edit))

    gold_all_edits = set()
    with open(gold_path) as f:
        for i, line in enumerate(f):
            line = line[:-1]
            if not line:
                continue
            for edit in line.split(' '):
                gold_all_edits.add((i, edit))

    num_true_positives = len(pred_all_edits & gold_all_edits)
    precision = num_true_positives / len(pred_all_edits)
    recall = num_true_positives / len(gold_all_edits)
    beta = 0.5
    if precision == 0.0 and recall == 0.0:
        f05 = 0.0
    else:
        f05 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
    print(f'precision = {precision}, recall = {recall}, f05 = {f05}')


if __name__ == '__main__':
    main()
