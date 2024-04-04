# run.sh

# 1. Run inference.
python inference.py \
    -i ./instruction_data \                          # Path to the evaluation data
    -f instruction_data_eval.json \                  # Name of the evaluation data file
    -o ./eval_output \                               # Path to the output directory
    -mp ./final_ckpt/<file_name> \                   # Path to the fine-tuned model checkpoint (Replace <file_name> with the file name of the fine-tuned model checkpoint. The file name has the format "model_YYYY-MM-DD-HHMMSS".)
    -mn meta-llama/Llama-2-7b-chat-hf \              # Name of the base model and tokenizer
    -c ../../../cache \                              # Path to the cache dir which saves the base model and tokenizer
    --seed 42 \                                      # Random seed
    --max_new_tokens 4096 \                          # Maximum number of tokens to generate (default: 4096)
    --do_sample False \                              # Whether to sample from the output distribution (default: False, i.e., greedy decoding)
    --temperature 1.0 \                              # Temperture value used to modulate the next token probabilities (default: 1.0)
    --top_k 50 \                                     # Number of highest probability vocabulary tokens to keep for top-k sampling (default: 50)
    --top_p 1.0 \                                    # If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for sampling (default: 1.0)
    --num_beams 1 \                                  # Number of beams for beam search (default: 1, i.e., greedy decoding, no beam search)
    --early_stopping False                           # Whether to stop generation when all beam hypotheses have reached the EOS token (default: False)

# 2. Run postprocessing to extract claim frames from the model output and store them to a tab-separated file.
python postprocessing.py -i ./eval_output -o ./claim_frames -d ../../../datasets

# Back to the root dir of this repo.
cd ../../../
pwd
ls