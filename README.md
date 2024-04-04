# CRUX-2023

skyline2023 at TAC 2023 CRUX.

## Setup

Prerequisite:
- python 3.11
- cuda 12.3

Install packages:
```bash
pip install -r requirements.txt
```
Download all the datasets to ```./datasets``` such that the directory has the following structure:  
*datasets*  
$\quad$|--*LDC*  
$\quad$| $\quad$|--*LDC2021E11*  
$\quad$| $\quad$|--*LDC2021E16*  
$\quad$| $\quad$|--*LDC2023E10*  
$\quad$|--*DWD_Overlay*  
$\quad$|--*CRUX2023_Task1_Evaluation_KBs*  
$\quad$|--*CRUX2023_Task1_Evaluation_root_uids.txt*  
$\quad$|--*CRUX2023_Task2_Evaluation_claim_frames.tab*  
$\quad$|--*CRUX2023_Evaluation_Topics.tab.txt*

## Data Preprocessing
Preprocess the raw datasets through the following steps:

1. Build mappings from the raw datasets by running:
  ```bash
  cd ./src/data_preprocessing/build_mappings/
  python build_mappings.py -i ../datasets -o ./output
  ```

2. Translate non-English segments to English by running:
  ```bash
  cd ./src/data_preprocessing/translate/
  python translate.py -i ../build_mappings/output -o ./output -m facebook/nllb-200-3.3B
  ```

3. Create translated documents (rsd files) by running:
  ```bash
  cd ./src/data_preprocessing/translate/

  # for trainval data
  python create_translated_rsd.py -i ./output/childuid2translatedsegments_trainval.p -o ./output

  # for eval data
  python create_translated_rsd.py -i ./output/childuid2translatedsegments_eval.p -o ./output
  ```

4. Create CoT instruction data by running:
  ```bash
  cd ./src/workshop/task1/

  # for trainval data (fine-tuning)
  python create_instruction_data_ft.py -dp ../../../datasets -trp ../../data_preprocessing/translate/output -o ./instruction_data

  # for eval data
  python create_instruction_data_eval.py -dp ../../../datasets -trp ../../data_preprocessing/translate/output -o ./instruction_data
  ```

5. Encode the instruction data for fine-tuning:
  ```bash
  cd ./src/workshop/task1/
  python encode.py -i ./instruction_data -o ./encoded_data -ofn train_val_1 -m meta-llama/Llama-2-7b-chat-hf
  ```

Alternatively, run the entire data preprocessing pipeline in one step:
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
```bash
   cd ./src/workshop/task1/
   
   python inference.py \
       -i ./instruction_data \                          # Path to the evaluation data
       -f instruction_data_eval.json \                  # Name of the evaluation data file
       -o ./eval_output \                               # Path to the output directory
       -mp ./final_ckpt/model_2024-04-04-075742 \       # Path to the fine-tuned model checkpoint
       -mn meta-llama/Llama-2-7b-chat-hf \              # Name of the model and tokenizer
       -c ../../../cache \                              # Path to the cache dir which saves the base model and tokenizer
       --seed 42 \                                      # Random seed
       --max_new_tokens 4096 \                          # Maximum number of tokens to generate (default: 4096)
       --do_sample False \                              # Whether to sample from the output distribution (default: False, i.e., greedy decoding)
       --temperature 1.0 \                              # Temperture value used to modulate the next token probabilities (default: 1.0)
       --top_k 50 \                                     # Number of highest probability vocabulary tokens to keep for top-k sampling (default: 50)
       --top_p 1.0 \                                    # If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for sampling (default: 1.0)
       --num_beams 1 \                                  # Number of beams for beam search (default: 1, i.e., greedy decoding, no beam search)
       --early_stopping False                           # Whether to stop generation when all beam hypotheses have reached the EOS token (default: False)
   ```

