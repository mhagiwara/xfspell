cat /path/to/fce-released-dataset/dataset/**/*.xml | python preprocessors/fce/generate_typo_pairs.py > data/fce/fce.tsv
cat data/fce/fce.tsv | python src/fce/split_sents.py > data/fce/fce-split.tsv
# there will be two errors in the file - we fixed them manually

cat data/fce/fce-split.tsv | cut -f1 | python src/normalize.py > data/fce/fce-split.norm.fr
cat data/fce/fce-split.tsv | cut -f2 | python src/normalize.py > data/fce/fce-split.norm.en

paste data/fce/fce-split.norm.fr data/fce/fce-split.norm.en | python src/extract_diffs.py > data/fce/preds/gold.diffs

cat data/fce/fce-split.norm.fr | python src/run_aspell.py > data/fce/preds/aspell
paste data/fce/fce-split.norm.fr data/fce/preds/aspell | python src/extract_diffs.py > data/fce/preds/aspell.diffs

