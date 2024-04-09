"""
Encode the instruction prompts by applying a sliding window and adding token_type_ids and labels.
"""

import torch, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain
from datasets import Dataset, DatasetDict
from transformers import LlamaTokenizer, AutoConfig
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence


device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
device_name = torch.cuda.get_device_name()
print(f"Using device: {device} ({device_name})")

# from huggingface_hub import notebook_login
# notebook_login()

def parse_arguments():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(description='Create translated rsd files.')
    parser.add_argument('-i', '--input_dir', type=str, default='./instruction_data', help="Path to the instruction prompts")
    parser.add_argument('-ifn', '--input_file_name', type=str, default='instruction_data_ft.json', help="Name of the input data file")
    parser.add_argument('-o', '--output_dir', type=str, default='./encoded_data', help="Path to the encoded data")
    parser.add_argument('-ofn', '--output_file_name', type=str, default='train_val_1', help="Specify the name of the output file")
    parser.add_argument('-m', '--model', type=str, default='meta-llama/Llama-2-7b-chat-hf', help="Specify the name of model and tokenizer")
    return parser.parse_args()


def mask_except_assist_answer(turn, assistant_id, mask_id):
    """
    Mask all the tokens in a turn except the answer of the assistant.
    
    Args:
        turn:list[int]    A list of input_ids in a turn.
        assistant_id:int  The id indicating the assistant (32001).
        
    Return:
        turn:list[int]    A list of input_ids in a turn with all the tokens masked except the answer of the assistant.
    """
    last_index = -1
    # Get the last index of the Assistant's answer in the turn by iterating over turn in reversed order until the assistant_id.
    for i in range(len(turn)-1, -1, -1):
        if turn[i] == assistant_id:
            last_index = 1
            break
    # Mask all the tokens in the turn except the Assistant's answer.
    for i in range(last_index):
        turn[i] = mask_id
    return turn

def apply_sliding_window(input_ids, token_type_ids, labels, max_len):
    """
    Apply a sliding window to slice the input_ids, token_type_ids and labels to the max length.
    
    Args:
        input_ids:list[int]
        token_type_ids:list[int]
        labels:list[int]
        max_len:int
        
    Return:
        input_ids_new:list[int]
        token_type_ids_new:list[int]
        labels_new:list[int]
    """
    print(len(input_ids), len(token_type_ids), len(labels))
    assert len(input_ids) == len(token_type_ids)
    assert len(token_type_ids) == len(labels)
    
    input_ids_new = []
    token_type_ids_new = []
    labels_new = []
    seg_id = 100
    
    for input_id, token_type_id, label in tqdm(zip(input_ids, token_type_ids, labels)):
        if len(input_id) <= max_len:
            input_ids_new.append([seg_id, input_id])
            token_type_ids_new.append([seg_id, token_type_id])
            labels_new.append([seg_id, label])
            continue
            
        # Initialize start index and end index.
        s_idx, e_idx = 0, max_len
        
        # Set step size.
        window_step_size = int(len(input_id) / max_len)
        #window_step_size = int(mean_dialogue_len)
        #window_step_size = int(0.5 * max_len)
        #window_step_size = int(0.25 * max_len)
        #window_step_size = int(0.75 * max_len)
        while True:
            input_id_sliced = input_id[s_idx: e_idx]
            token_type_id_sliced = token_type_id[s_idx: e_idx]
            label_sliced = label[s_idx: e_idx]
            
            input_ids_new.append([seg_id, input_id_sliced])
            token_type_ids_new.append([seg_id, token_type_id_sliced])
            labels_new.append([seg_id, label_sliced])
            
            if e_idx >= len(input_id):
                break
            s_idx += window_step_size
            e_idx += window_step_size
        seg_id += 1
    return input_ids_new, token_type_ids_new, labels_new


def main():
    args = parse_arguments()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Input directory:", input_dir)
    print("Output directory:", output_dir)
    print("Model and tokenizer:", args.model)
    
    # Load Llama 2 tokenizer.
    tokenizer = LlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model,
        cache_dir='../../../cache/',
        add_eos_token=True
    )
    
    model_config = AutoConfig.from_pretrained(args.model)
    print("Maximal input length:", model_config.max_position_embeddings)
    
    special_tokens = {
        'additional_special_tokens': [
            '<User>',
            '<Assistant>',
            '<MASK>',
            '<<SYS>>',
            '<</SYS>>',
            '[INST]',
            '[/INST]'
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    vocab = tokenizer.get_vocab()
    print("Vocab size:", len(vocab))
    
    # Load instruction prompts.
    dialogues_df = pd.read_json(f"{input_dir}/{args.input_file_name}")
    
    dialogue_list = []
    for conversation in dialogues_df['conversations']:
        for dialogue in conversation:
            dialogue_list.append(dialogue['content'])
            
    # Compute the average token length of the dialogue.
    dialogue_lengths = []
    for dialogue in dialogue_list:
        dialogue = dialogue.strip()
        ids = tokenizer.encode(dialogue)
        dialogue_lengths.append(len(ids))
    avg_dialogue_len = np.mean(dialogue_lengths)
    print("Average token length:", avg_dialogue_len)
    
    # Input length (#tokens) sys+User+Assistant
    input_length = []
    for dialogue in dialogue_list:
        encoded_input = tokenizer.encode(dialogue, return_tensors='pt')
        input_length.append(encoded_input.shape[1])
        
    # Compute average input length, variance and standard deviation.
    input_length = np.asarray(input_length)
    print("Average length:", input_length.mean())
    print("Variance:", input_length.var())
    print("Standard deviation:", input_length.std())
    # Maximal and minimal input lengths.
    print("Maximum:", input_length.max())
    print("Minimum:", input_length.min())
    
    # Count number of dialogues that exceed the maximal acceptable length.
    num_out_of_max_length = 0
    for dialogue_len in dialogue_lengths:
        if dialogue_len > 4096:
            num_out_of_max_length += 1
    print(f"Number of dialogues that exceed the maximal acceptable length ({model_config.max_position_embeddings}):", num_out_of_max_length, f", i.e. {num_out_of_max_length / len(dialogue_lengths)}")
    
    
    # Get ids of special tokens.
    bos_id = vocab['<s>']
    eos_id = vocab['</s>']
    user_id = vocab['<User>']
    assistant_id = vocab['<Assistant>']
    mask_id = vocab['<mask>']
    bosys_id = vocab['<<SYS>>']
    eosys_id = vocab['<</SYS>>']
    boinst_id = vocab['[INST]']
    eoinst_id = vocab['[/INST]']
    
    print('bos_id:', bos_id)
    print('eos_id:', eos_id)
    print('user_id:', user_id)
    print('assistant_id:', assistant_id)
    print('mask_id:', mask_id)
    print('bosys_id:', bosys_id)
    print('eosys_id:', eosys_id)
    print('boinst_id:', boinst_id)
    print('eoinst_id:', eoinst_id)
    
    # Each element in cleaned_dialogues is a list containing a (question/instruction, answer)-pair, where the question/instruction is stated by the user and the answer is generated by the assistant.
    cleaned_dialogues = []
    qa = []
    for i, dialogues in enumerate(dialogue_list):
        qa.append(dialogues)
        if i % 2 == 1:
            cleaned_dialogues.append(qa)
            qa = []
            
            
    # Get token_ids.
    token_ids = []
    for dialogue in tqdm(cleaned_dialogues):
        dialogue_ids = []
        for utterance in dialogue:
            # Special tokens have been added manually.
            ids = tokenizer.encode(utterance, add_special_tokens=False)
            dialogue_ids.append(ids)
        token_ids.append(dialogue_ids)
        
    # Convert the token_ids into dialogues with speaker_ids.
    # Since the speaker_ids have been already added during the instruction prompt creation, we don't need to do anything here.
    dialogues_with_speaker_ids = token_ids
    
    # Create input_ids for each turn in each dialogue (input_ids already contain bos_ids and eos_ids).
    input_ids = []
    for dialogue in tqdm(dialogues_with_speaker_ids):
        input_ids.append(list(chain.from_iterable(dialogue)))
        
    num_oom_input_ids = 0
    for i, input_id in enumerate(input_ids):
        if len(input_id) > model_config.max_position_embeddings:
            num_oom_input_ids += 1
    print(f"Number of input_ids longer than {model_config.max_position_embeddings}:", num_oom_input_ids, f", i.e. {num_oom_input_ids / len(input_ids)}")
    
    # Create token_type_ids for each turn in each dialogue.
    # The token_type_ids component indicates the speaker of each segment in the input_ids.
    # It only includes the ids of the User (user_id:  32000) and Assistant (assistant_id: 32001) tokens.
    token_type_ids = []
    for turn in tqdm(input_ids):
        turn_token_type_ids = []
        # Initialize type_id as user_id.
        type_id = user_id
        for token_id in turn:
            # If the current token_id is identical to the user_id, just add it to the token type id list of the turn.
            if token_id == user_id:
                turn_token_type_ids.append(type_id)
            # If the current token_id is identical to the assistant_id, change the value of type_id to that of assistant_id and add it to the token type id list of the turn.
            elif token_id == assistant_id:
                type_id = assistant_id
                turn_token_type_ids.append(type_id)
        token_type_ids.append(turn_token_type_ids)
    
    # Create labels for each turn in each dialogue by masking out all the tokens except for the answer of the assistant.
    labels = []
    for turn in tqdm(input_ids):
        turn_labels = mask_except_assist_answer(turn, assistant_id, mask_id)
        labels.append(turn_labels)
        
    # Slice the data using a sliding window.
    input_ids_sliced, token_type_ids_sliced, labels_sliced = apply_sliding_window(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        labels=labels,
        max_len=model_config.max_position_embeddings
    )
    
    print(len(input_ids_sliced))
    print(len(token_type_ids_sliced))
    print(len(labels_sliced))
    
    # Create segment ids for each turn and convert them to tensor.
    seg_ids_tensor = []
    for input_id_turn in tqdm(input_ids_sliced):
        seg_ids_tensor.append(input_id_turn[0])
    seg_ids_tensor = torch.LongTensor(seg_ids_tensor)
    print(f"seg_ids_tensor shape: {seg_ids_tensor.shape}")
    
    # Create input ids for each turn and convert them to tensor.
    input_ids_tensor = []
    for input_id_turn in tqdm(input_ids_sliced):
        input_ids_tensor.append(torch.LongTensor(input_id_turn[1]))
    # Pad the input_ids_tensor with eos_id (2).
    # The padding length is the length of the longest utterance in the dialogue.
    input_ids_tensor = pad_sequence(
        input_ids_tensor,
        batch_first=True,
        padding_value=eos_id
    )
    print(f"input_ids_tensor shape: {input_ids_tensor.shape}")
    
    # Create token type ids for each turn and convert them to tensor.
    token_type_ids_tensor = []
    for token_type_id_turn in tqdm(token_type_ids_sliced):
        token_type_ids_tensor.append(torch.LongTensor(token_type_id_turn[1]))
    # Pad the token_type_ids_tensor with eos_id (2).
    # The padding length is the length of the longest utterance in the dialogue.
    token_type_ids_tensor = pad_sequence(
        token_type_ids_tensor,
        batch_first=True,
        padding_value=eos_id
    )
    print(f"token_type_ids_tensor: {token_type_ids_tensor.shape}")
    
    # Create labels for each turn and convert them to tensor.
    labels_tensor = []
    for label_turn in tqdm(labels_sliced):
        labels_tensor.append(torch.LongTensor(label_turn[1]))
    # Pad the labels_tensor with eos_id (2).
    # The padding length is the length of the longest utterance in the dialogue.
    labels_tensor = pad_sequence(
        labels_tensor,
        batch_first=True,
        padding_value=eos_id
    )
    print(f"labels_tensor: {labels_tensor.shape}")
    
    
    data_dict = {
        'seg_ids': seg_ids_tensor,
        'input_ids': input_ids_tensor,
        'token_type_ids': token_type_ids_tensor,
        'labels': labels_tensor
    }
    
    hf_dataset = Dataset.from_dict(data_dict)
    
    # Split the dataset.
    train_val = hf_dataset.train_test_split(shuffle=True, seed=123, test_size=0.2)
    
    train_val = DatasetDict({
        'train': train_val['train'],
        'validation': train_val['test']
    })
    
    train_val.save_to_disk(f"{output_dir}/{args.output_file_name}")


if __name__ == "__main__":
    main()
    