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

# create bt512 dataset from
cat /path/to/w2c.txt /path/to/tatoeba.txt | python src/format_monolingual.py | sort | uniq > data/bt/bt512.txt

cat data/bt/bt512.txt | python src/normalize.py > data/bt/bt512.norm.txt

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
    bin/gtc-bt512-filtered \
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
    --save-dir models/bt02 \
    --log-format json --log-interval 10 \
    --max-epoch 40 \
    | tee logs/bt02.log

cat data/fce/fce-split.norm.fr | python src/tokenize.py \
    | fairseq-interactive \
    bin/gtc-bt512-filtered \
    --path models/bt02/checkpoint_best.pt \
    --source-lang fr --target-lang en \
    --beam 10 --max-len-a 1.2 --max-len-b 0 \
    | python src/format_fairseq_output.py > data/fce/preds/bt02

paste data/fce/fce-split.norm.fr data/fce/preds/bt01 | python src/extract_diffs.py > data/fce/preds/bt01.diffs

# bt02 - add upper-cased data

cat data/bt/gtc-bt512.filtered.tok.en \
  | awk 'NR%10==0' \
  | tr '[:lower:]' '[:upper:]' \
  > data/bt/gtc-bt512.filtered.mod10-0.upper.tok.en

cat data/bt/gtc-bt512.filtered.tok.fr \
  | awk 'NR%10==0' \
  | tr '[:lower:]' '[:upper:]' \
  > data/bt/gtc-bt512.filtered.mod10-0.upper.tok.fr

cat data/bt/gtc-bt512.filtered.tok.en data/bt/gtc-bt512.filtered.mod10-0.upper.tok.en > data/bt/gtc-bt512.filtered.upper.tok.en
cat data/bt/gtc-bt512.filtered.tok.fr data/bt/gtc-bt512.filtered.mod10-0.upper.tok.fr > data/bt/gtc-bt512.filtered.upper.tok.fr

fairseq-preprocess --source-lang fr --target-lang en \
    --trainpref data/bt/gtc-bt512.filtered.upper.tok \
    --validpref data/gtc/dev.tok \
    --destdir bin/gtc-bt512-filtered-upper

fairseq-train \
    bin/gtc-bt512-filtered-upper \
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
    --save-dir models/bt03 \
    --log-format json --log-interval 10 \
    --max-epoch 40 \
    | tee logs/bt03.log

cat data/fce/fce-split.norm.fr | head -n 2500 | python src/tokenize.py \
    | fairseq-interactive \
    bin/gtc-bt512-filtered-upper \
    --path models/bt03/checkpoint_best.pt \
    --source-lang fr --target-lang en \
    --beam 10 --max-len-a 1.2 --max-len-b 0 \
    | python src/format_fairseq_output.py > data/fce/preds/bt03-1

# create owt100 dataset (first 100 files from Open Web Text corpus)
ls data/openwebtext/*.xz | xargs -IH tar -C data/openwebtext/text/ -xvf H
find 'data/openwebtext/text/' -name '*.txt' | xargs cat | python src/bt/format_monolingual.py > data/bt/owt100.txt

cat data/bt/owt100.txt | python src/normalize.py > data/bt/owt100.norm.txt

cat data/bt/owt100.norm.txt \
    | awk 'NR%4==0' \
    | python src/tokenize.py \
    | fairseq-interactive \
    bin/gtc-en2fr \
    --path models/gtc-en2fr/checkpoint_best.pt \
    --source-lang en --target-lang fr \
    --beam 1 --sampling --sampling-topk 5 --max-len-a 1 --max-len-b 200 \
    --buffer-size 100 --max-tokens 4096 2> /dev/null \
    | python src/format_fairseq_output.py > data/bt/owt100.mod4-0.fr-pred &

for i in `seq 0 3`;
do
    cat data/bt/owt100.norm.txt \
       | awk 'NR%4=='"$i"'' \
     >> data/bt/owt100.en
done

for i in `seq 0 3`;
do
    cat data/bt/owt100.mod4-$i.fr-pred \
     >> data/bt/owt100.fr
done

cat data/bt/bt512.en | python src/tokenize.py > data/bt/bt512.tok.en
cat data/bt/bt512.fr | python src/tokenize.py > data/bt/bt512.tok.fr

cat data/bt/owt100.en | python src/tokenize.py > data/bt/owt100.tok.en
cat data/bt/owt100.fr | python src/tokenize.py > data/bt/owt100.tok.fr

# size
#    242,304 data/gtc/train.tok.en
#  1,407,728 data/bt/bt512.tok.en
#    573,817 data/bt/owt100.tok.en
#  2,223,849 total
cat data/gtc/train.tok.en data/bt/bt512.tok.en data/bt/owt100.tok.en > data/bt/gtc-bt512-owt100.tok.en
cat data/gtc/train.tok.fr data/bt/bt512.tok.fr data/bt/owt100.tok.fr > data/bt/gtc-bt512-owt100.tok.fr

cat data/bt/gtc-bt512-owt100.tok.en \
  | awk 'NR%10==0' \
  | tr '[:lower:]' '[:upper:]' \
  > data/bt/gtc-bt512-owt100.mod10-0.upper.tok.en

cat data/bt/gtc-bt512-owt100.tok.fr \
  | awk 'NR%10==0' \
  | tr '[:lower:]' '[:upper:]' \
  > data/bt/gtc-bt512-owt100.mod10-0.upper.tok.fr

cat data/bt/gtc-bt512-owt100.tok.en data/bt/gtc-bt512-owt100.mod10-0.upper.tok.en > data/bt/gtc-bt512-owt100.upper.tok.en
cat data/bt/gtc-bt512-owt100.tok.fr data/bt/gtc-bt512-owt100.mod10-0.upper.tok.fr > data/bt/gtc-bt512-owt100.upper.tok.fr

fairseq-preprocess --source-lang fr --target-lang en \
    --trainpref data/bt/gtc-bt512-owt100.upper.tok \
    --validpref data/gtc/dev.tok \
    --destdir bin/gtc-bt512-owt100-upper

fairseq-train \
    bin/gtc-bt512-owt100-upper \
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
    --save-dir models/bt04 \
    --log-format json --log-interval 10 \
    --max-epoch 40 \
    | tee logs/bt04.log


# create owt1k dataset (first 1/100th from Open Web Text corpus)
# fist, copy openwebtext/urlsf_subset00-*_data.xz under data/openwebtext
ls data/openwebtext/*.xz | xargs -IH tar -C data/openwebtext/text/ -xvf H
find 'data/openwebtext/text/' -name '*.txt' | xargs cat | python src/bt/format_monolingual.py > data/bt/owt1k.txt

cat data/bt/owt1k.txt | python src/normalize.py > data/bt/owt1k.norm.txt

cat data/bt/owt1k.norm.txt \
    | awk 'NR%16==0' \
    | python src/tokenize.py \
    | fairseq-interactive \
    bin/gtc-en2fr \
    --path models/gtc-en2fr/checkpoint_best.pt \
    --source-lang en --target-lang fr \
    --beam 1 --sampling --sampling-topk 5 --max-len-a 1 --max-len-b 200 \
    --buffer-size 100 --max-tokens 4096 2> /dev/null \
    | python src/format_fairseq_output.py > data/bt/owt1k.mod16-0.fr-pred &

for i in `seq 0 15`;
do
    cat data/bt/owt1k.norm.txt \
       | awk 'NR%16=='"$i"'' \
     >> data/bt/owt1k.en
done

for i in `seq 0 15`;
do
    cat data/bt/owt1k.mod16-$i.fr-pred \
     >> data/bt/owt1k.fr
done

# size
#    242,304 data/gtc/train.tok.en
#  1,407,728 data/bt/bt512.tok.en
#  5,853,841 data/bt/owt1k.tok.en
#  7,503,873 total
cat data/bt/owt1k.en | python src/tokenize.py > data/bt/owt1k.tok.en
cat data/bt/owt1k.fr | python src/tokenize.py > data/bt/owt1k.tok.fr

cat data/gtc/train.tok.en data/bt/bt512.tok.en data/bt/owt1k.tok.en > data/bt/gtc-bt512-owt1k.tok.en
cat data/gtc/train.tok.fr data/bt/bt512.tok.fr data/bt/owt1k.tok.fr > data/bt/gtc-bt512-owt1k.tok.fr

cat data/bt/gtc-bt512-owt1k.tok.en \
  | awk 'NR%10==0' \
  | tr '[:lower:]' '[:upper:]' \
  > data/bt/gtc-bt512-owt1k.mod10-0.upper.tok.en

cat data/bt/gtc-bt512-owt1k.tok.fr \
  | awk 'NR%10==0' \
  | tr '[:lower:]' '[:upper:]' \
  > data/bt/gtc-bt512-owt1k.mod10-0.upper.tok.fr

cat data/bt/gtc-bt512-owt1k.tok.en data/bt/gtc-bt512-owt1k.mod10-0.upper.tok.en > data/bt/gtc-bt512-owt1k.upper.tok.en
cat data/bt/gtc-bt512-owt1k.tok.fr data/bt/gtc-bt512-owt1k.mod10-0.upper.tok.fr > data/bt/gtc-bt512-owt1k.upper.tok.fr

fairseq-preprocess --source-lang fr --target-lang en \
    --trainpref data/bt/gtc-bt512-owt1k.upper.tok \
    --validpref data/gtc/dev.tok \
    --destdir bin/gtc-bt512-owt1k-upper \
    --workers 4

fairseq-train \
    bin/gtc-bt512-owt1k-upper \
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
    --save-dir models/bt05 \
    --log-format json --log-interval 10 \
    --max-epoch 40 \
    | tee logs/bt05.log

cat data/fce/fce-split.norm.fr | python src/tokenize.py \
    | fairseq-interactive \
    bin/gtc \
    --path models/gtc01/checkpoint_best.pt \
    --source-lang fr --target-lang en \
    --beam 10 --max-len-a 1 --max-len-b 200 \
    | python src/format_fairseq_output.py > data/fce/preds/gtc01

cat data/fce/fce-split.us.norm.fr | python src/tokenize.py \
    | fairseq-interactive \
    bin/gtc-bt512-owt1k-upper \
    --path models/bt05/checkpoint_best.pt \
    --source-lang fr --target-lang en \
    --beam 10 --max-len-a 1.2 --max-len-b 0 \
    | python src/format_fairseq_output.py > data/fce/preds/bt05

echo "tisimptant too spll chck ths dcment." \
    | python src/tokenize.py \
    | fairseq-interactive \
    bin/gtc-bt512-owt1k-upper \
    --path models/bt05/checkpoint_best.pt \
    --source-lang fr --target-lang en \
    --beam 10 --max-len-a 1 --max-len-b 200

echo "The book Tom and Jerry put on the yellow desk yesterday wer about NLP." \
    | python src/tokenize.py \
    | fairseq-interactive \
    bin/gtc-bt512-owt1k-upper \
    --path models/bt05/checkpoint_best.pt \
    --source-lang fr --target-lang en --beam 10

echo "The books Tom and Jerry put on the yellow desk yesterday wer about NLP." \
    | python src/tokenize.py \
    | fairseq-interactive \
    bin/gtc-bt512-owt1k-upper \
    --path models/bt05/checkpoint_best.pt \
    --source-lang fr --target-lang en --beam 10
