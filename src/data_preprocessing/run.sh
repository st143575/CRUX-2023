# run.sh

# 1. Build mappings from the raw datasets.
cd ./build_mappings/
python build_mappings.py -i ../../datasets -o ./output

# 2. Translate non-English segments to English.
cd ../translate/
python translate.py -i ../build_mappings/output -o ./output -m facebook/nllb-200-3.3B

# 3. Create translated documents (rsd files).
# for trainval data
python create_translated_rsd.py -i ./output/childuid2translatedsegments_trainval.p -o ./output
# for eval data
python create_translated_rsd.py -i ./output/childuid2translatedsegments_eval.p -o ./output

# 4. Create CoT instruction data.
cd ../../workshop/task1/
# for trainval data (fine-tuning)
python create_instruction_data_ft.py -dp ../../../datasets -trp ../../data_preprocessing/translate/output -o ./instruction_data
# for eval data
python create_instruction_data_eval.py -dp ../../../datasets -trp ../../data_preprocessing/translate/output -o ./instruction_data

# 5. Encode the instruction data for fine-tuning.
python encode.py -i ./instruction_data -o ./encoded_data -ofn train_val_1 -m meta-llama/Llama-2-7b-chat-hf

# Back to the root dir.
cd ../../../