# run.sh

# 1. Run inference.
python inference.py -i ./instruction_data -f instruction_data_eval.json -o ./eval_output -mp ./final_ckpt/model_2024-04-04-075742 -mn meta-llama/Llama-2-7b-chat-hf -c ../../../cache --seed 42 --max_new_tokens 4096 --do_sample --temperature 0.7 --top_k 50 --top_p 1.0 --num_beams 3

# 2. Run postprocessing to extract claim frames from the model output and store them to a tab-separated file.
python postprocessing.py -i ./eval_output -o ./claim_frames -d ../../../datasets

# Back to the root dir of this repo.
cd ../../../