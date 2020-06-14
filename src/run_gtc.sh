cat /path/to/github-typo-corpus.v1.0.0.jsonl.gz | gzip -d | python src/gtc/format_gtc.py > data/gtc/all.tsv

cat data/gtc/all.tsv | cut -f1 | python src/tokenize.py > data/gtc/all.tok.fr
cat data/gtc/all.tsv | cut -f2 | python src/tokenize.py > data/gtc/all.tok.en

cat data/gtc/all.tok.fr | awk 'NR%20!=0' > data/gtc/train.tok.fr
cat data/gtc/all.tok.fr | awk 'NR%20==0' > data/gtc/dev.tok.fr

cat data/gtc/all.tok.en | awk 'NR%20!=0' > data/gtc/train.tok.en
cat data/gtc/all.tok.en | awk 'NR%20==0' > data/gtc/dev.tok.en

fairseq-preprocess --source-lang fr --target-lang en \
    --trainpref data/gtc/train.tok \
    --validpref data/gtc/dev.tok \
    --destdir bin/gtc

fairseq-train \
    bin/gtc \
    --fp16 \
    --arch transformer \
    --encoder-layers 6 --decoder-layers 6 \
    --encoder-embed-dim 1024 --decoder-embed-dim 1024 \
    --encoder-ffn-embed-dim 4096 --decoder-ffn-embed-dim 4096 \
    --encoder-attention-heads 16 --decoder-attention-heads 16 \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.997)' --adam-eps 1e-09 --clip-norm 25.0 \
    --lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 16000 \
    --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
    --weight-decay 0.00025 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --max-tokens 4096 \
    --save-dir models/gtc01 \
    --log-format json --log-interval 10 \
    --max-epoch 40 \
    | tee logs/gtc-01.log

cat data/fce/fce-split.norm.fr | python src/tokenize.py \
    | fairseq-interactive \
    bin/gtc \
    --path models/gtc01/checkpoint_best.pt \
    --source-lang fr --target-lang en \
    --beam 10 --max-len-a 1 --max-len-b 200 \
    | python src/format_fairseq_output.py > data/fce/preds/gtc01

echo "tisimptant too spll chck ths dcment." \
    | python src/tokenize.py \
    | fairseq-interactive \
    bin/gtc \
    --path models/gtc01/checkpoint_best.pt \
    --source-lang fr --target-lang en \
    --beam 10 --max-len-a 1 --max-len-b 200
