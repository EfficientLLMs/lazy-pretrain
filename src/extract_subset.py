import json
from transformers import AutoTokenizer
from tqdm import tqdm

def extract_tokens(input_file, output_file, target_token_count):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    
    total_tokens = 0
    extracted_data = []

    # Open the input file and read line by line
    with open(input_file, 'r') as infile:
        for line in tqdm(infile, desc="Processing lines"):
            data = json.loads(line)
            text = data['text']
            
            tokens = tokenizer.encode(text)
            token_count = len(tokens)
            total_tokens += token_count
            extracted_data.append(data)
            
            if total_tokens >= target_token_count:
                break
    
    print(f"Extracted {total_tokens} tokens")
    
    with open(output_file, 'w') as outfile:
        for item in extracted_data:
            json.dump(item, outfile)
            outfile.write('\n')

if __name__ == "__main__":
    input_file = "data/train_00/00.jsonl"
    output_file = "data/train_00/00_20m.jsonl"
    target_tokens = 20_000_000
    
    extract_tokens(input_file, output_file, target_tokens)
    print(f"Extracted data written to {output_file}")