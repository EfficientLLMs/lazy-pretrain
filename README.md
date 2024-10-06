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

## Download and use the pre-tokenized data with the same order
EleutherAI has provided a pre-tokenized version of the standard (duplicated) pile dataset, which is also Pythia pre-shuffled. The dataset contains only token_ids. [link](https://huggingface.co/datasets/EleutherAI/pile-standard-pythia-preshuffled/tree/main)

The whole dataset has about 300B tokens. `00.bin` to `19.bin` are about 30GB large each. The last one `20.bin` is only 78.3MB. We can download only the last one.

1. Clone the repository without downloading
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/EleutherAI/pile-standard-pythia-preshuffled
```
2. Download only the last file, which has 39168000 tokens
```bash
cd pile-standard-pythia-preshuffled
git lfs pull --include="document-00020-of-00020.bin"
```
1. Get all the tokens from the last file. We can use it to get the last 5M tokens.
```python
filename = "document-00020-of-00020.bin"
tokens = np.memmap(filename, dtype=np.uint16)
```
