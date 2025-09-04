#!/bin/bash

# iterate over all models in models.txt
while read -r model; do
    # if the model starts with a #, skip it
    if [[ $model == "#"* ]]; then
        continue
    fi
    # if the model is empty, skip it
    if [[ -z $model ]]; then
        continue
    fi
    echo "Testing $model"
    python test_accuracy.py --questions data/faith_benchmark_questions.json --model $model
done < models.txt
