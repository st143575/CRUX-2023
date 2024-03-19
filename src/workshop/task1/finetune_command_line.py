import random, time, torch, argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from datasets import load_from_disk
from transformers import (
    LlamaTokenizer, 
    LlamaForCausalLM,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from peft import (
    prepare_model_for_kbit_training, 
    LoraConfig, 
    get_peft_model
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create translated rsd files.')
    parser.add_argument('-i', '--input_dir', type=str, default='./encoded_data/train_val_1', help="Path to the encoded data for finetuning")
    parser.add_argument('-o', '--output_dir', type=str, default='./ckpts', help="Path to model checkpoints")
    parser.add_argument('-m', '--model', type=str, default='meta-llama/Llama-2-7b-chat-hf', help="Name of the model and tokenizer")
    # Arguments for training hyperparameters.
    parser.add_argument('-e', '--num_epochs', type=int, default=1, help="Number of epochs")
    parser.add_argument('-bs', '--batch_size', type=int, default=2, help="Batch size")
    parser.add_argument('--optimizer', type=str, default='paged_adamw_8bit', help="Name of optimizer")
    parser.add_argument('--warm_up_steps', type=int, default=10, help="Warm up steps")
    parser.add_argument('--weight_decay', type=float, default=0.1, help="Weight decay")
    parser.add_argument('--logging_steps', type=int, default=64, help="Logging steps")
    parser.add_argument('--save_steps', type=int, default=512, help="Save steps")
    parser.add_argument('--save_total_limit', type=int, default=2, help="Save total limit")
    # Arguments for QLoRA.
    parser.add_argument('-r', '--rank', type=int, default=8, help="")
    parser.add_argument('-a', '--lora_alpha', type=int, default=32, help="")
    parser.add_argument('-d', '--lora_dropout', type=float, default=0.05, help="")
    parser.add_argument('-b', '--lora_bias', type=str, default='none', help="")
    return parser.parse_args()


def finetune(
        dataset,
        model,
        tokenizer,
        data_collator,
        output_dir,
        num_epochs,
        batch_size,
        optimizer,
        warm_up_steps,
        weight_decay,
        logging_steps,
        save_steps,
        save_total_limit,
        ):
    torch.cuda.empty_cache()
    training_args = TrainingArguments(
        output_dir=output_dir,
    #     overwrite_output_dir=True,
        num_train_epochs=num_epochs,
    #     optim='adamw_torch',
        optim=optimizer,
        save_strategy='steps',
        evaluation_strategy='epoch',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warm_up_steps,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_dir='./logs/',
        fp16=True,
    )
    
    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=dataset['train'], 
        eval_dataset=dataset['validation'], 
        data_collator=data_collator,
    )
    
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    time_in_seconds = end_time - start_time
    time_in_hours = time_in_seconds / 3600
    print(f"Training time: {time_in_seconds} seconds, i.e. {time_in_hours} hours.")
    
    # Save fine-tuned model and tokenizer.
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    model_ckpt_name = f"model_{timestamp}"
    model.save_pretrained(f"./final_ckpt/{model_ckpt_name}")
    tokenizer.save_pretrained(f"./final_ckpt/{model_ckpt_name}")
    print(f"Model and tokenizer are saved to ./final_ckpt/{model_ckpt_name}.")


def main():
    device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
    device_name = torch.cuda.get_device_name()
    print(f"Using device: {device} ({device_name})")
    
    # Set seeds for reproducibility.
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # from huggingface_hub import notebook_login
    # notebook_login()

    args = parse_arguments()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Input directory:", input_dir)
    print("Output directory:", output_dir)
    
    # Load dataset.
    dataset = load_from_disk(f'{input_dir}')
    
    # Load the default data collator.
    default_data_collator = DefaultDataCollator(return_tensors='pt')
    
    # Load tokenizer and model.
    tokenizer = LlamaTokenizer.from_pretrained(args.model, cache_dir='../../../cache/', add_eos_token=True)
    model = LlamaForCausalLM.from_pretrained(args.model, cache_dir='../../../cache/').to(device)
    
    special_tokens = {
        'additional_special_tokens': ['<User>', '<Assistant>', '<MASK>', '<<SYS>>', '<</SYS>>', '[INST]', '[/INST]']
    }
    tokenizer.add_special_tokens(special_tokens)
    vocab = tokenizer.get_vocab()
    model.resize_token_embeddings(len(vocab))
    
    # Quantize the model and set the configurations for QLoRA.
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=args.rank, 
        lora_alpha=args.lora_alpha, 
        target_modules=[
        "q_proj",
        "up_proj",
        "o_proj",
        "k_proj",
        "down_proj",
        "gate_proj",
        "v_proj"], 
        lora_dropout=args.lora_dropout, 
        bias=args.lora_bias, 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)

    finetune(
        dataset=dataset, 
        model=model, 
        tokenizer=tokenizer, 
        data_collator=default_data_collator,
        output_dir=output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        warm_up_steps=args.warm_up_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
    )
    
    print("Done!")


if __name__ == "__main__":
    main()