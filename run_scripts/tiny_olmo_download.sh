#!/bin/bash

MODEL_NAME="700M"
STEP="step406934"

# Make directory for model
mkdir "models/tiny-olmo-"$MODEL_NAME"-"$STEP"-unsharded"
cd "models/tiny-olmo-"$MODEL_NAME"-"$STEP"-unsharded"


for filename in "model.pt" "config.yaml" "train.pt";
do
    if [[ "$MODEL_NAME" == "700M" ]]; then
        wget https://olmo-checkpoints.org/checkpoints/olmo-tiny/tiny-olmo-$MODEL_NAME-rms-norm-adam-eps-1e-8-emb-wd/$STEP-unsharded/$filename
    else    
        wget https://olmo-checkpoints.org/checkpoints/olmo-tiny/tiny-olmo-$MODEL_NAME-rms-norm-adam-eps-1e-8-lr-6e-4-emb-wd/$STEP-unsharded/$filename
    fi
done
