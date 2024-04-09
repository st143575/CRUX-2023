import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
import torch, json, argparse
import numpy as np
import dill as pickle
from tqdm import tqdm
from pathlib import Path
from itertools import chain
from datasets import Dataset, DatasetDict
from transformers import LlamaTokenizer, AutoConfig
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
    parser.add_argument('-ifn', '--input_file_name', type=str, default='instruction_data_ft.p', help="Name of the input data file")
    parser.add_argument('-o', '--output_dir', type=str, default='./encoded_data', help="Path to the encoded data")
    parser.add_argument('-ofn', '--output_file_name', type=str, default='train_val_1', help="Name of the output data file")
    parser.add_argument('-m', '--model', type=str, default='meta-llama/Llama-2-7b-chat-hf', help="Name of model and tokenizer")
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


def main():
    args = parse_arguments()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Load tokenizer.
    tokenizer = LlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model, 
        cache_dir='../../../cache/', 
        add_eos_tokens=True
        )
    tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(args.model)
    print("Maximal input length:", model_config.max_position_embeddings)

    # Load customized special tokens.
    with open('./special_tokens_task2.json', 'r') as file:
        special_tokens = json.load(file)

    USER = special_tokens['USER']
    ASSISTANT = special_tokens['ASSISTANT']
    B_INST = special_tokens['B_INST']
    E_INST = special_tokens['E_INST']
    B_SYS = special_tokens['B_SYS']
    E_SYS = special_tokens['E_SYS']
    MASK = special_tokens['MASK']
    print("Special tokens:")
    print(USER)
    print(ASSISTANT)
    print(B_INST)
    print(E_INST)
    print(B_SYS)
    print(E_SYS)
    print(MASK)
        
    # Add customized special tokens to the tokenizer.
    additional_special_tokens = {
        'additional_special_tokens': [USER, ASSISTANT, B_INST, E_INST, B_SYS, E_SYS, MASK]
    }
    tokenizer.add_special_tokens(additional_special_tokens)
    vocab = tokenizer.get_vocab()
    print(len(vocab))

    # Load instruction data.
    instruction_data = pickle.load(open(f"{input_dir}/{args.input_file_name}", "rb"))

    # Check input lengths.
    dialogue_lengths = []
    for ids, dialogue, in tqdm(instruction_data.items()):
        qa = dialogue['User'] + dialogue['Assistant']
        encoded_input = tokenizer.encode(qa)
        dialogue_lengths.append(len(encoded_input))
            
    # Average input length, variance and standard deviation.
    dialogue_lengths = np.array(dialogue_lengths)
    print(f"Average dialogue length: {np.mean(dialogue_lengths)}")
    print(f"Variance: {np.var(dialogue_lengths)}")
    print(f"Standard deviation: {np.std(dialogue_lengths)}")
    print(f"Maximum: {np.max(dialogue_lengths)}")
    print(f"Minimum: {np.min(dialogue_lengths)}")

    # Count number of dialogues that exceed the maximal acceptable length.
    num_out_of_max_length = 0
    for dialogue_len in dialogue_lengths:
        if dialogue_len > 4096:
            num_out_of_max_length += 1
    print(f"Number of dialogues that exceed the maximal acceptable length ({model_config.max_position_embeddings}):", num_out_of_max_length, f", i.e. {num_out_of_max_length / len(dialogue_lengths)}")
    print(f"Since no dialogue exceeds the maximal acceptable length of the model {args.model}, there is no need to use the sliding window.")

    # Get ids of special tokens.
    bos_id = vocab['<s>']
    eos_id = vocab['</s>']
    user_id = vocab['<User>']
    assistant_id = vocab['<Assistant>']
    mask_id = vocab['<MASK>']
    bosys_id = vocab['<<SYS>>\n']
    eosys_id = vocab['\n<</SYS>>\n\n']
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

    # Get token_ids.
    token_ids = []
    for ids, dialogue in tqdm(instruction_data.items()):
        dialogue_ids = []
        for speaker, utterance in dialogue.items():
            encoded_input = tokenizer.encode(
                utterance,
                add_special_tokens=False
            )
            dialogue_ids.append(encoded_input)
        token_ids.append(dialogue_ids)
    
    # Create input_ids for each turn in each dialogue.
    # input_ids already contains bos_ids and eos_ids.
    input_ids = []
    for dialogue in tqdm(token_ids):
        input_ids.append(list(chain.from_iterable(dialogue)))
    print("InputID length:", len(input_ids))

    # Create token_type_ids for each turn in each dialogue.
    # The token_type_ids indicate the speaker of each segment in the input_ids sequence.
    token_type_ids = []
    for turn in tqdm(input_ids):
        turn_token_type_ids = []
        type_id = user_id
        for token_id in turn:
            if token_id == user_id:
                type_id = user_id
                turn_token_type_ids.append(type_id)
            elif token_id == assistant_id:
                type_id = assistant_id
                turn_token_type_ids.append(type_id)
            else:
                turn_token_type_ids.append(type_id)
        token_type_ids.append(turn_token_type_ids)
    print("Token type ID length:", len(token_type_ids))

    # Create labels for each turn in each dialogue by masking out all the tokens except for the answer of the assistant.
    labels = []
    for turn in tqdm(input_ids):
        turn_labels = mask_except_assist_answer(turn, assistant_id, mask_id)
        labels.append(turn_labels)
    print("Label length:", len(labels))

    # Create input ids for each turn and convert them to tensor.
    input_ids_tensor = []
    for input_id_turn in tqdm(input_ids):
        input_ids_tensor.append(torch.LongTensor(input_id_turn))
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
    for token_type_id_turn in tqdm(token_type_ids):
        token_type_ids_tensor.append(torch.LongTensor(token_type_id_turn))
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
    for label_turn in tqdm(labels):
        labels_tensor.append(torch.LongTensor(label_turn))
    # Pad the labels_tensor with eos_id (2).
    # The padding length is the length of the longest utterance in the dialogue.
    labels_tensor = pad_sequence(
        labels_tensor,
        batch_first=True,
        padding_value=eos_id
    )
    print(f"labels_tensor: {labels_tensor.shape}")

    data_dict = {
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

if __name__ == '__main__':
    main()