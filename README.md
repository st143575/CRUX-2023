# CRUX-2023

skyline2023 at [TAC 2023 CRUX](https://tac.nist.gov/2023/KBP/CRUX/index.html).

## Setup

Prerequisite:
- python 3.11
- cuda 12.3

Install packages:
```bash
pip install -r requirements.txt
```
Download all the datasets to ```./datasets``` such that the directory has the following structure:  
*datasets/*  
$\quad$|--*LDC/*  
$\quad$| $\quad$|--*LDC2021E11/*  
$\quad$| $\quad$|--*LDC2021E16/*  
$\quad$| $\quad$|--*LDC2023E10/*  
$\quad$|--*DWD_Overlay/*  
$\quad$|--*CRUX2023_Task1_Evaluation_KBs/*  
$\quad$|--*CRUX2023_Task1_Evaluation_root_uids.txt*  
$\quad$|--*CRUX2023_Task2_Evaluation_claim_frames.tab*  
$\quad$|--*CRUX2023_Evaluation_Topics.tab.txt*

## Data Preprocessing
Preprocess the raw datasets through the following steps:

1. Build mappings from the raw datasets:
  ```bash
  cd ./src/data_preprocessing/build_mappings/
  python build_mappings.py -i ../datasets -o ./output
  ```

2. Translate non-English segments to English:
  ```bash
  cd ./src/data_preprocessing/translate/
  python translate.py -i ../build_mappings/output -o ./output -m facebook/nllb-200-3.3B
  ```

3. Create translated documents (rsd files):
  ```bash
  cd ./src/data_preprocessing/translate/

  # for trainval data
  python create_translated_rsd.py -i ./output/childuid2translatedsegments_trainval.p -o ./output

  # for eval data
  python create_translated_rsd.py -i ./output/childuid2translatedsegments_eval.p -o ./output
  ```

4. Create and encode CoT instruction data (Only training data need to be encoded in advance.):
  ```bash
  # Task 1
  cd ./src/workshop/task1/

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
  ```

Alternatively, run the entire data preprocessing pipeline in one line:
  ```bash
  bash ./src/data_preprocessing/run.sh
  ```

## Task 1
### Fine-tuning
   ```bash
   cd ./src/workshop/task1/
   
   python finetune.py \
       -i ./encoded_data \                      # Path to the encoded data for finetuning
       -dfn train_val_1 \                       # Name of the dataset file
       -o ./ckpts \                             # Path to model checkpoints
       -m meta-llama/Llama-2-7b-chat-hf \       # Name of the model and tokenizer
       -e 1 \                                   # Number of epochs
       -bs 2 \                                  # Batch size (default: 2, largest possible batch size for a single RTX A6000: 8)
       --optimizer paged_adamw_8bit \           # Optimizer
       --warm_up_steps 10 \                     # Warm up steps
       --weight_decay 0.1 \                     # Weight decay
       --logging_steps 64 \                     # Number of steps for which the trainer generates logs
       --save_steps 512 \                       # Number of steps for which the trainer saves a model checkpoint
       --save_total_limit 2 \                   # Maximal number of model checkpoints saved
       -r 8 \                                   # LoRA rank parameter (LoRA attention dimension)
       -a 32 \                                  # The alpha parameter for Lora scaling
       -d 0.05 \                                # The dropout probability for Lora layers
       -b 'none'                                # Bias type for LoRA (default: do not update biases during fine-tuning)
   ```

The fine-tuning took about 6 days on a single RTX A6000.

### Inference
1. Run inference to obtain output files containing model-generated claim frames expressed in natural language:
 ```bash
 cd ./src/workshop/task1/
 
 python inference.py \
     -i ./instruction_data \                          # Path to the evaluation data
     -f instruction_data_eval.json \                  # Name of the evaluation data file
     -o ./eval_output \                               # Path to the output directory
     -mp ./final_ckpt/<file_name> \                   # Path to the fine-tuned model checkpoint (Replace <file_name> with the file name of the fine-tuned model checkpoint. The file name has the format "model_YYYY-MM-DD-HHMMSS".)
     -mn meta-llama/Llama-2-7b-chat-hf \              # Name of the base model and tokenizer
     -c ../../../cache \                              # Path to the cache dir which saves the base model and tokenizer
     --seed 42 \                                      # Random seed
     --max_new_tokens 4096 \                          # Maximum number of tokens to generate (default: 4096)
     --do_sample \                                    # Whether to sample from the output distribution (default: False, i.e., greedy decoding)
     --temperature 0.7 \                              # Temperture value used to modulate the next token probabilities (default: 1.0)
     --top_k 50 \                                     # Number of highest probability vocabulary tokens to keep for top-k sampling (default: 50)
     --top_p 1.0 \                                    # If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for sampling (default: 1.0)
     --num_beams 3 \                                  # Number of beams for beam search (default: 1, i.e., greedy decoding, no beam search)
 ```

2. Post-process the generated output files to extract structured claim frames in tab-separated format:
 ```bash
 python postprocess.py -i ./eval_output -o ./claim_frames
 ```

Alternatively, run the entire inference pipeline in one line:
  ```bash
  bash ./src/workshop/task1/run.sh
  ```


## Task 2
### Fine-tuning
  ```bash
     cd ./src/workshop/task2/
     
     python finetune.py \
         -i ./encoded_data \                      # Path to the encoded data for finetuning
         -dfn train_val_1 \                       # Name of the dataset file
         -o ./ckpts \                             # Path to model checkpoints
         -m meta-llama/Llama-2-7b-chat-hf \       # Name of the model and tokenizer
         -e 1 \                                   # Number of epochs
         -bs 2 \                                  # Batch size (default: 2, largest possible batch size for a single RTX A6000: 8)
         --optimizer paged_adamw_8bit \           # Optimizer
         --warm_up_steps 10 \                     # Warm up steps
         --weight_decay 0.1 \                     # Weight decay
         --logging_steps 64 \                     # Number of steps for which the trainer generates logs
         --save_steps 512 \                       # Number of steps for which the trainer saves a model checkpoint
         --save_total_limit 2 \                   # Maximal number of model checkpoints saved
         -r 8 \                                   # LoRA rank parameter (LoRA attention dimension)
         -a 32 \                                  # The alpha parameter for Lora scaling
         -d 0.05 \                                # The dropout probability for Lora layers
         -b 'none'                                # Bias type for LoRA (default: do not update biases during fine-tuning)
  ```

The fine-tuning took about 3.6 hours on a single RTX A6000.

### Inference
1. Run inference to obtain output files containing model-generated answers expressed in natural language:
  ```bash
  cd ./src/workshop/task2/

  python inference.py \
      -i ./instruction_data \
      -f instruction_data_eval.p \
      -o ./eval_output \
      -mp ./final_ckpt/<file_name> \            # Path to the fine-tuned model checkpoint (Replace <file_name> with the file name of the fine-tuned model checkpoint. The file name has the format "model_YYYY-MM-DD-HHMMSS".)
      -mn meta-llama/Llama-2-7b-chat-hf \
      -c ../../../cache \
      --seed 42 \
      --min_new_tokens 200 \
      --max_new_tokens 1000 \
      --do_sample \
      --temperature 1.0 \
      --top_k 10 \
      --top_p 1.0 \
      --num_beams 3 \
      --early_stopping
  ```

2. Post-process the generated output files to extract cross-claim relations in tab-separated format:
  ```bash
  python postprocessing.py -i ./eval_output -o ./cross_claim_relations
  ```

Alternatively, run the entire inference pipeline in one line:
  ```bash
  bash ./src/workshop/task2/run.sh
  ```
