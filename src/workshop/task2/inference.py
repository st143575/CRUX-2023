import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
import torch, json, argparse, random, time
import numpy as np
import dill as pickle
from tqdm import tqdm
from pathlib import Path
from peft import PeftModelForCausalLM
from transformers import (
    LlamaTokenizer, 
    LlamaForCausalLM, 
    BitsAndBytesConfig
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate outputs for all test instances and save each output to a separate file.')
    parser.add_argument('-i', '--input_dir', type=str, default='./instruction_data', help="Path to the evaluation data")
    parser.add_argument('-f', '--file_name', type=str, default='instruction_data_eval.p', help="Name of the evaluation data file")
    parser.add_argument('-o', '--output_dir', type=str, default='./eval_output', help="Path to the output directory")
    parser.add_argument('-mp', '--model_path', type=str, default='./final_ckpt/model_2024-04-09-030150', help="Path to the fine-tuned model checkpoint")
    parser.add_argument('-mn', '--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf', help="Name of the base model and tokenizer")
    parser.add_argument('-c', '--cache_dir', type=str, default='../../../cache', help="Path to the cache dir which saves the base model and tokenizer")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    # Arguments for generation hyperparameters.
    parser.add_argument('--min_new_tokens', type=int, default=200, help="Minimum number of tokens to generate (default: 200)")
    parser.add_argument('--max_new_tokens', type=int, default=1000, help="Maximum number of tokens to generate (default: 1000)")
    parser.add_argument('--do_sample', action='store_true', default=False, help="Whether to sample from the output distribution (default: False, i.e., greedy decoding)")
    parser.add_argument('--temperature', type=float, default=1.0, help="Temperture value used to modulate the next token probabilities (default: 1.0)")
    parser.add_argument('--top_k', type=int, default=10, help="Number of highest probability vocabulary tokens to keep for top-k sampling (default: 50)")
    parser.add_argument('--top_p', type=float, default=1.0, help="If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for sampling (default: 1.0)")
    parser.add_argument('--num_beams', type=int, default=3, help="Number of beams for beam search (If set to 1: greedy decoding, no beam search)")
    parser.add_argument('--early_stopping', action='store_true', default=False, help="Whether to stop generation when all beam hypotheses have reached the EOS token (default: False)")
    return parser.parse_args()


def load_model_and_tokenizer(model_name, base_model_dir, adapter_dir, device):
    """
    Load the tokenizer and model with adapter.
    The tokenizer is the fine-tuned tokenizer comprising the pre-trained tokenizer on HuggingFace with task-specific config.
    The model is the instruction fine-tuned model consisting of the pre-trained model and the fine-tuned low-rank adapter.
    Since only the adapter layers are saved after the fine-tuning, we have to initialize a new base model and then
    attach the adapter layers to it.
    """

    # Set configuration for quantization.
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the tokenizer.
    # Set cache_dir to the directory where the pre-trained model is saved to load the default tokenizer.
    # Set pretrained_model_name_or_path to the directory where the fine-tuned model is saved to.
    tokenizer = LlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path=adapter_dir, 
        cache_dir=base_model_dir, 
        model_max_length=4096, 
        attention_mask=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    print(tokenizer)

    # Load the base model in 4 bit.
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        cache_dir=base_model_dir,
        load_in_4bit=True,
        quantization_config=quant_config
    )
    print(model.num_parameters())

    # Resize model's token embeddings to match the tokenizer.
    model.resize_token_embeddings(len(tokenizer))

    # Load and add the fine-tuned adapter layers to the base model.
    model = PeftModelForCausalLM.from_pretrained(model=model, model_id=adapter_dir, device=device)
    print(model.num_parameters())

    return model, tokenizer


def predict(
        prompt,
        tokenizer,
        model,
        device,
        min_new_tokens,
        max_new_tokens,
        do_sample,
        temperature,
        top_k,
        top_p,
        num_beams,
        early_stopping
):
    """
    Generate output for a given test instance.
    """
    
    input_ids = tokenizer.encode(
        prompt,
        return_tensors='pt',
        truncation=True,
        add_special_tokens=False
    ).to(device)
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_beams=num_beams,
        early_stopping=early_stopping,
    )
    output = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    return output
    

def predict_and_save(
        test_set,
        tokenizer,
        model,
        device,
        output_dir,
        min_new_tokens,
        max_new_tokens,
        do_sample,
        temperature,
        top_k,
        top_p,
        num_beams,
        early_stopping
):
    """
    Generate outputs for all test instances and save each output to a separate file.
    """
    time_per_instance = []
    for ids, dialogue in tqdm(test_set.items()):
        doc_id_1 = ids[0]
        claim_id_1 = ids[1]
        doc_id_2 = ids[2]
        claim_id_2 = ids[3]
        prompt = f"{dialogue['User']}\n{dialogue['Assistant']}"
        start_time = time.time()
        output = predict(
            prompt,
            tokenizer,
            model,
            device,
            min_new_tokens,
            max_new_tokens,
            do_sample,
            temperature,
            top_k,
            top_p,
            num_beams,
            early_stopping
        )
        end_time = time.time()
        time_in_seconds = end_time - start_time
        time_in_minutes = time_in_seconds / 60
        time_per_instance.append(time_in_minutes)
        print(f"Inference time for instance {ids}: {time_in_minutes} minutes.")

        with open(f"{output_dir}/{doc_id_1}_{claim_id_1}_{doc_id_2}_{claim_id_2}.txt", "w") as f:
            f.write(output)

    time_per_instance = np.array(time_per_instance)
    avg_time_per_instance = np.mean(time_per_instance)
    with open(f"./inference_time_per_instance.txt", "w") as f:
        f.write(f"Average inference time per instance: {avg_time_per_instance} minutes.")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name()
    print(f"Using device: {device} ({device_name})")

    args = parse_arguments()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path)
    cache_dir = Path(args.cache_dir)
    print("Input directory:", input_dir)
    print("Input file name:", args.file_name)
    print("Output directory:", output_dir)
    print("Model path:", model_path)
    print("Model name:", args.model_name)
    print("Cache directory:", cache_dir)

    min_new_tokens = args.min_new_tokens
    max_new_tokens = args.max_new_tokens
    do_sample = args.do_sample
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    num_beams = args.num_beams
    early_stopping = args.early_stopping
    print("Generation hyperparameters:")
    print(f"max new tokens: {max_new_tokens}")
    print(f"do_sample: {do_sample}")
    print(f"temperature: {temperature}")
    print(f"top_k: {top_k}")
    print(f"top_p: {top_p}")
    print(f"num_beams: {num_beams}")
    print(f"early_stopping: {early_stopping}")

    # Set seeds for reproducibility.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        base_model_dir=cache_dir,
        adapter_dir=model_path,
        device=device
    )
    model.eval()
    

    # Load the evaluation dataset.
    with open(f"{input_dir}/{args.file_name}", 'rb') as file:
        instruction_data_eval = pickle.load(file)
    print(f"The evaluation data has {len(instruction_data_eval)} instances.\n")
    print("An example instance:", list(instruction_data_eval.items())[0])

    # Generate outputs for all test instances and save each output to a separate file.
    predict_and_save(
        test_set=instruction_data_eval,
        tokenizer=tokenizer,
        model=model,
        device=device,
        output_dir=output_dir,
        min_new_tokens=min_new_tokens,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_beams=num_beams,
        early_stopping=early_stopping
    )
    print("Done!")


if __name__ == "__main__":
    main()