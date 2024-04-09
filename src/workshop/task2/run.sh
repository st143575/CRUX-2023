# run.sh

# 1. Run inference.
python inference.py -i ./instruction_data -f instruction_data_eval.p -o ./eval_output -mp ./final_ckpt/model_2024-04-09-030150 -mn meta-llama/Llama-2-7b-chat-hf--do_sample -c ../../../cache --seed 42 --min_new_tokens 200 --max_new_tokens 1000 --do_sample --temperature 1.0 --top_k 10 --top_p 1.0 --num_beams 3 --early_stopping

# 2. Run postprocessing to extract cross-claim relations from the model output files and store them to a tab-separated file.
python postprocessing.py -i ./eval_output -o ../cross_claim_relations

# Back to the root dir of this repo.
cd ../../../