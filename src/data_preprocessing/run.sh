### Datapreprocessing run.sh ###

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

# 4. Create and encode CoT instruction data. Only training data need to be encoded in advance.
# Task 1
cd ../../workshop/task1/
# Create trainval data
python create_instruction_data_ft.py -dp ../../../datasets -trp ../../data_preprocessing/translate/output -o ./instruction_data
python encode.py -i ./instruction_data -ifn instruction_data_ft.json -o ./encoded_data -ofn train_val_1 -m meta-llama/Llama-2-7b-chat-hf
# Create eval data
python create_instruction_data_eval.py -dp ../../../datasets -trp ../../data_preprocessing/translate/output -o ./instruction_data

# Task 2
cd ../task2/
# Create trainval data
python create_instruction_data_ft.py -i ../../../datasets -o ./instruction_data
python encode.py -i ./instruction_data -ifn instruction_data_ft.p -o ./encoded_data -ofn train_val_1 -m meta-llama/Llama-2-7b-chat-hf
# Create eval data
python create_instruction_data_eval.py -i ../../../datasets -o ./instruction_data


# Back to the root dir of this repo.
cd ../../../