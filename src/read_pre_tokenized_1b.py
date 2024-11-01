import numpy as np
import torch

def prepare_chunks(tokens, seq_len):
    chunks = []

    for i in range(0, len(tokens), seq_len):
        chunk = tokens[i:i+seq_len]

        if len(chunk) == seq_len:
            chunks.append(chunk)

    chunks = torch.tensor(chunks, dtype=torch.long)

    return chunks


def main():
    file1 = "data/pile-standard-pythia-preshuffled/document-00020-of-00020.bin"
    file2 = "data/pile-standard-pythia-preshuffled/document-00019-of-00020.bin"
    total_tokens = 1_000_000_000
    seq_len = 512

    tokens1 = np.memmap(file1, dtype=np.uint16)

    if len(tokens1) > total_tokens:
        tokens = tokens1[:total_tokens]

    else:
        tokens2 = np.memmap(file2, dtype=np.uint16)[-(total_tokens - len(tokens1)):]

        print(f"Number of tokens in file 1: {len(tokens1)}")
        print(f"Number of tokens in file 2: {tokens2.shape}")

        tokens = np.concatenate([tokens1, tokens2])

    print(f"Number of tokens: {len(tokens)}")

    # Chunk and save as torch tensors
    chunks = prepare_chunks(tokens, seq_len)

    


if __name__ == '__main__':
    main()