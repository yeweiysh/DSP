# Distributional Shortest-Path Graph Kernels 

This is a PyTorch implementation of "Distributional Shortest-Path Graph Kernels" submitted to IEEE TKDE.

## Requirements
See `requriements.txt` for required Python libraries.

Test on a server with a dual-core Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz, 256 GB, NVIDIA RTX 3090 GPU, 24GB, and Ubuntu 18.04.6 LTS operating system.

## Train from Scratch

### Train Tokenizer
Run the following command in the terminal to train a tokenizer:
```
python tokenizer.py --dataset MUTAG --depth 7
```

### Train Model

Run the following command in the terminal to train a Transformer model:
```
python train.py --dataset MUTAG --depth 7 --batch_size 1024 --epochs 200 --d_model 16 --lr 0.001
```
The model is saved in "./model/MUTAG/best_model.pkl".

### Save Shortest-Path Embeddings (Optional)

Run the following command in the terminal to save the embeddings of the shortest paths:
```
python save_sp_emb.py --dataset MUTAG --depth 7 --tokenizer_file ./tokenizer/MUTAG/depth_7_tokenizer.json --model_file ./model/MUTAG/best_model.pkl
```
The embeddings are saved in "./sp_embs/MUTAG/".

### Graph Classification

Run the following command in the terminal to perform graph classification:
```
python eval.py --dataset MUTAG --depth 2 --t 50 --c 4 --load_embedding
```

### Quick Start

Just run the following command in the terminal to obtain the results in the paper:
```
bash best_res.sh
```
