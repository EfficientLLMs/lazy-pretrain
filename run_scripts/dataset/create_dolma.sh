#!/bin/bash
#SBATCH --job-name=create_dolma
#SBATCH --mem=32G
#SBATCH --output=output_logs/create_dolma.out
#SBATCH --error=error_logs/create_dolma.err
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu

DATA_DIR='data/dolma'
PARALLEL_DOWNLOADS='10'
DOLMA_VERSION='v1_7'

# Clone the repository if it doesn't already exist
if [ ! -d "dolma" ]; then
    git clone git@hf.co:datasets/allenai/dolma
fi

mkdir -p "${DATA_DIR}"

# Download only the first 10 files of 103
echo "Downloading the first 10 files from Dolma"

head -n 10 "dolma/urls/${DOLMA_VERSION}.txt" | xargs -n 1 -P "${PARALLEL_DOWNLOADS}" wget -q -P "$DATA_DIR"

echo "Finished downloading the first 10 files from Dolma. Load data using the following code:"
echo ""
echo "from datasets import load_dataset"
echo "file_pattern = 'dolma-v1_7*.json.gz'"
echo "dataset = load_dataset('json', data_files=f'{args.data_dir}/{file_pattern}', split='train', streaming=True)"
