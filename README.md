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
3. Get all the tokens from the last file. We can use it to get the last 5M tokens.
```python
filename = "document-00020-of-00020.bin"
tokens = np.memmap(filename, dtype=np.uint16)
```

## Grow model from small size to large size

Run the following command to grow the model from `pythia-410m` to `pythia-1.4b`. 

```bash
python src/grow.py \
    --small_model "pythia-410m" \
    --large_depth 24 \
    --large_width 2048 \
    --depth_growth "alternate" \
    --attn_heads 16 \
    --output_dir "models/pythia-410m-to-pythia-1.4b"
```

Run the following command to grow the model from `pythia-70m` to `pythia-410m`. Note that since the number of attention heads is not specified, the value will be calculated based on the width of the model.

```bash
python src/grow.py \
    --small_model "pythia-70m" \
    --large_depth 24 \
    --large_width 1024 \
    --depth_growth "alternate" \
    --output_dir "models/pythia-70m-to-pythia-410m"
```

More details about the model architecture can be found in the [github repository](https://github.com/EleutherAI/pythia/tree/main?tab=readme-ov-file#models) for Pythia.


## Train the grown model with the first 20M tokens using LoRA

Ensure that you have extracted the first 20M tokens from the training data. The file should be saved in `data/train_00/00_20m.jsonl`.

1. First, pre-tokenize the data. This will save the tokenized data in a `.pt` file under the specified directory.

```bash
python src/prepare_dataset.py \
    --dataset "data/train_00/00_20m.jsonl" \
    --tokenizer "EleutherAI/pythia-70m" \
    --output "data/train_00/00_20m.pt"
```

2. Train the model with the first 20M tokens using LoRA. The model will be saved in the specified directory.

```bash
accelerate launch src/pretrain_lora.py \
    --grown_model "models/pythia-410m-to-pythia-1.4b" \
    --tokenizer "EleutherAI/pythia-410m" \
    --dataset "data/train_00/00_20m.pt" \
    --seed 1234 \
    --rank 256 \
    --batch_size 8 \
    --lr 1e-4 \
    --output_dir "models/pythia-410m-to-pythia-1.4b-lora"
```

You can run this command using `sbatch` if by running the following command. Ensure that you modify the script to include your email ID.

```bash
sbatch run_scripts/pretrain_lora.sh
```

## Evaluate the model using lm-eval

To evaluate the model, you must install the `lm-eval` package. Go to the directory in which you want to install the package and run the following command.

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

After installing the package, you can evaluate the model using the following command. You can specify the tasks you want to evaluate the model on along with the paths to the model, tokenizer, and the results file. Also specify the path to the file containing your Hugging Face API token (`.token`). The script will read the first line of the file to get the token.

```bash
python src/eval_llm.py \
    --base_model_path "models/pythia-410m-to-pythia-1.4b" \
    --lora_path "models/pythia-410m-to-pythia-1.4b-lora-5m" \
    --tokenizer_path "EleutherAI/pythia-410m" \
    --eval_results_path "eval/r256_410m_1.4b_eval_results" \
    --parallelize \
    --tasks "lambada_openai" "arc_easy" "arc_challenge" "hellaswag" "piqa" "winogrande" "sciq" "logiqa" "logiqa2" "openbookqa" \
    --batch_size 8 \
    --token ".token"
```

You can run this command using `sbatch` if by running the following command. Ensure that you modify the script to include your email ID.

```bash
sbatch run_scripts/eval_llm.sh
```
