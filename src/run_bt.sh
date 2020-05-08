# train a corrupter
fairseq-preprocess --source-lang en --target-lang fr \
    --trainpref data/gtc/train.tok \
    --validpref data/gtc/dev.tok \
    --destdir bin/gtc-en2fr

fairseq-train \
    bin/gtc-en2fr \
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
    --save-dir models/gtc-en2fr \
    --log-format json --log-interval 10 \
    --max-epoch 40 \
    | tee logs/gtc-en2fr01.log

echo 'The quick brown fox jumps over the lazy dog.' | python src/tokenize.py \
    | fairseq-interactive \
    bin/gtc-en2fr \
    --path models/gtc-en2fr/checkpoint_best.pt \
    --source-lang en --target-lang fr \
    --beam 1 --sampling --sampling-topk 10 --max-len-a 1 --max-len-b 200 \
    | python src/format_fairseq_output.py

cat data/bt/bt512.norm.txt \
    | awk 'NR%20==4' \
    | python src/tokenize.py \
    | fairseq-interactive \
    bin/gtc-en2fr \
    --path models/gtc-en2fr/checkpoint_best.pt \
    --source-lang en --target-lang fr \
    --beam 1 --sampling --sampling-topk 5 --max-len-a 1 --max-len-b 200 \
    --buffer-size 100 --max-tokens 4096 2> /dev/null \
    | python src/format_fairseq_output.py > data/bt/bt512.mod20-4.fr-pred &

for i in `seq 0 19`;
do
    cat data/bt/bt512.norm.txt \
       | awk 'NR%20=='"$i"'' \
     >> data/bt/bt512.en
done

for i in `seq 0 19`;
do
    cat data/bt/bt512.mod20-$i.fr-pred \
     >> data/bt/bt512.fr
done

paste data/bt/bt512.en data/bt/bt512.fr \
  | python src/bt/filter_bitext.py \
  | cut -f1 \
  | python src/tokenize.py \
  > data/bt/bt512.filtered.tok.en

paste data/bt/bt512.en data/bt/bt512.fr \
  | python src/bt/filter_bitext.py \
  | cut -f2 \
  | python src/tokenize.py \
  > data/bt/bt512.filtered.tok.fr

cat data/gtc/train.tok.en data/bt/bt512.filtered.tok.en > data/bt/gtc-bt512.filtered.tok.en
cat data/gtc/train.tok.fr data/bt/bt512.filtered.tok.fr > data/bt/gtc-bt512.filtered.tok.fr

fairseq-preprocess --source-lang fr --target-lang en \
    --trainpref data/bt/gtc-bt512.filtered.tok \
    --validpref data/gtc/dev.tok \
    --destdir bin/gtc-bt512-filtered

fairseq-train \
    bin/gtc-bt512 \
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
    --save-dir models/bt01 \
    --log-format json --log-interval 10 \
    --max-epoch 40 \
    | tee logs/bt01.log

cat data/fce/fce-split.norm.fr | python src/tokenize.py \
    | fairseq-interactive \
    bin/gtc-bt512 \
    --path models/bt01/checkpoint_best.pt \
    --source-lang fr --target-lang en \
    --beam 10 --max-len-a 1.2 --max-len-b 0 \
    | python src/format_fairseq_output.py > data/fce/preds/bt01

paste data/fce/fce-split.norm.fr data/fce/preds/bt01 | python src/extract_diffs.py > data/fce/preds/bt01.diffs
