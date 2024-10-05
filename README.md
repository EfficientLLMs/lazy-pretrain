# Lazy Pretrain

## Extract first 5M tokens from the training data
1. Decompress the first subset of the training data
```bash
pv data/the_pile/train/00.jsonl.zst | zstd -d > data/train_00/00.jsonl
```
2. (Optional) Check the first 5 lines and keys of each line
```bash
head -n 5 data/train_00/00.jsonl
head -n 1 data/train_00/00.jsonl | jq 'keys'
```
3. Extract the first 5M tokens (2723 lines). The result will be saved in `data/train_00/00_5m.jsonl`
```bash
python extract_subset.py
```