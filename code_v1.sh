#!/bin/bash

device=cpu
dataset=rlm
mode=last
num_features=64
num_layers=1
num_tokens=2
beta=2
train_size=$((2**18))
batch_size=128
accumulation=1
test_size=$((2**15))
input_format=long
whitening=0
model=transformer_mlm
depth=1
embedding_dim=64
num_heads=1
ffwd_size=4
dropout=0.0
optim=adam
lr=1e-3
momentum=0.0
scheduler=cosine-warmup
warmup_time=8
decay_time=$((2**18))
max_epochs=8
print_freq=1024
save_freq=2
measure_train=TRUE

root="/Users/julialand/Desktop/SISSA/RLM/"
proj="RLM_results"

# Bucle para correr el experimento 8 veces
for i in {1..8}; do
  echo "Ejecutando experimento $i..."

  # Cambiar nombre de salida en cada iteración
  outname="${root}/${proj}/${dataset}${num_features}L${num_layers}_b${beta}_ts${train_size}_$i"

  # Nuevas semillas en cada iteración
  seed_rules=$(od -An -N3 -tu4 /dev/urandom | tr -d '[:space:]')
  seed_sample=$(od -An -N3 -tu4 /dev/urandom | tr -d '[:space:]')
  seed_model=$(od -An -N3 -tu4 /dev/urandom | tr -d '[:space:]')

  # Ejecutar el script de Python
  python RLM_clean/main.py \
    --device "$device" \
    --dataset "$dataset" \
    --mode "$mode" \
    --num_features "$num_features" \
    --num_layers "$num_layers" \
    --seed_rules "$seed_rules" \
    --num_tokens "$num_tokens" \
    --beta "$beta" \
    --train_size "$train_size" \
    --batch_size "$batch_size" \
    --accumulation "$accumulation" \
    --test_size "$test_size" \
    --seed_sample "$seed_sample" \
    --input_format "$input_format" \
    --whitening "$whitening" \
    --model "$model" \
    --depth "$depth" \
    --embedding_dim "$embedding_dim" \
    --num_heads "$num_heads" \
    --ffwd_size "$ffwd_size" \
    --dropout "$dropout" \
    --seed_model "$seed_model" \
    --optim "$optim" \
    --lr "$lr" \
    --momentum "$momentum" \
    --scheduler "$scheduler" \
    --warmup_time "$warmup_time" \
    --decay_time "$decay_time" \
    --max_epochs "$max_epochs" \
    --print_freq "$print_freq" \
    --save_freq "$save_freq" \
    --measure_train \
    --outname "$outname"
done
