"""
Translate non-English segments in the LTF files to English ones.
"""

import os, torch, argparse
import dill as pickle
from tqdm import tqdm
from pathlib import Path
from collections import Counter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
device_name = torch.cuda.get_device_name()
print(f"Using device: {device} ({device_name})")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Build mappings between datasets.')
    parser.add_argument('-i', '--input_dir', type=str, default='../build_mappings/output', help="Path to the mappings created")
    parser.add_argument('-o', '--output_dir', type=str, default='./output', help="Path to save the translated texts")
    parser.add_argument('-m', '--model', type=str, default='facebook/nllb-200-3.3B', help="The translator model (default: facebook/nllb-200-3.3B)")
    return parser.parse_args()


def check_doclang(input_file):
    """Get language statistics of the documents."""
    childuid2doclang = pickle.load(open(input_file, 'rb'))
    doclang_counter = Counter(childuid2doclang.values())
    print("Count document language:\n", doclang_counter, "\n")
    
    conclusion = """There are 483 segments absolutely in English, 395 segments absolutely in Spanish and 248 segments absolutely in Russian. However, there are 13 segments in mixed languages, i.e. 10 in English and Spanish, 2 in English and Russian, 1 in Russian and Ukrainian.
    When translating them to English, we ideally have to consider Spanish, Russian and Ukrainian as non-English source languages. But in practice, we assume that 'eng,spa' only contain Spanish texts, 'eng,rus' only contain Russian texts, and 'rus,ukr' only contain Russian texts."""
    print(conclusion)
    return childuid2doclang
    
    
def translate_segment(inp, tokenizer, model, device, target_lang='eng_Latn'):
    inputs = tokenizer(inp, return_tensors="pt").to(device)
    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang], max_length=100)
    output = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return output
    
    
def translate(childuid2segments, childuid2doclang, tokenizers, model, device):
    """
    Translate non-English segments in the LTF files to English ones.
    
    Model: NLLB
    Page: https://huggingface.co/facebook/nllb-200-3.3B
    Doc: https://huggingface.co/docs/transformers/main/en/model_doc/nllb#overview-of-nllb
    List of languages and Flores-200 code: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
    """
    childuid2translatedsegments = dict()
    for child_uid, segments in tqdm(childuid2segments.items()):
        # print(child_uid, segments)
        segmentid2translatedsegment = dict()
        
        # If the document language is English, we don't need to translate it. Just copy the original segments to childuid2translatedsegments.
        if childuid2doclang[child_uid] == 'eng':
            childuid2translatedsegments[child_uid] = segments

        # If the document language is not English, use different tokenizers for different source languages.
        else:
            # Russian
            if childuid2doclang[child_uid] == 'rus':
                for segment_id, segment in segments.items():
                    # translated_segment
                    segmentid2translatedsegment[segment_id] = translate_segment(segment, tokenizer=tokenizers['tokenizer_rus'], model=model, device=device)
            # Spanish
            elif childuid2doclang[child_uid] == 'spa':
                for segment_id, segment in segments.items():
                    # translated_segment
                    segmentid2translatedsegment[segment_id] = translate_segment(segment, tokenizer=tokenizers['tokenizer_spa'], model=model, device=device)
            # English and Spanish
            elif childuid2doclang[child_uid] == 'eng,spa':
                # Assume 'eng,spa' only contain Spanish texts!
                for segment_id, segment in segments.items():
                    # translated_segment
                    segmentid2translatedsegment[segment_id] = translate_segment(segment, tokenizer=tokenizers['tokenizer_spa'], model=model, device=device)
            # English and Russian
            elif childuid2doclang[child_uid] == 'eng,rus':
                # Assume 'eng,rus' only contain Russian texts!
                for segment_id, segment in segments.items():
                    # translated_segment
                    segmentid2translatedsegment[segment_id] = translate_segment(segment, tokenizer=tokenizers['tokenizer_rus'], model=model, device=device)
            # Russian and Ukrainian
            elif childuid2doclang[child_uid] == 'rus,ukr':
                # Assume 'rus,ukr' only contain Russian texts!
                for segment_id, segment in segments.items():
                    # translated_segment
                    segmentid2translatedsegment[segment_id] = translate_segment(segment, tokenizer=tokenizers['tokenizer_rus'], model=model, device=device)
            # Other languages are considered invalid.
            else:
                raise ValueError("Invalid document language!")
            childuid2translatedsegments[child_uid] = segmentid2translatedsegment
    
    print("Translation done!")
    return childuid2translatedsegments


def main():
    args = parse_arguments()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    translator_model_name = args.model
    print("Input directory:", input_dir)
    print("Output directory:", output_dir)
    print("Translator model:", translator_model_name)
    
    # Get language statistics of the documents.
    print("Document language of train_val set:")
    childuid2doclang_trainval = check_doclang(f'{input_dir}/childuid2doclang_trainval.p')
    print("Document language of eval set:")
    childuid2doclang_eval = check_doclang(f'{input_dir}/childuid2doclang_eval.p')
    
    # Load the mapping from child_uid to segments in original languages (childuid2segments_trainval.p).
    # (train_val)
    childuid2segments_trainval = pickle.load(open(f'{input_dir}/childuid2segments_trainval.p', 'rb'))
    print(len(childuid2segments_trainval))

    # Load the mapping from child_uid to segments in original languages (childuid2segments_trainval.p).
    # (eval)
    childuid2segments_eval = pickle.load(open(f'{input_dir}/childuid2segments_eval.p', 'rb'))
    print(len(childuid2segments_eval))
    
    # Load model.
    model = AutoModelForSeq2SeqLM.from_pretrained(translator_model_name, token=True, cache_dir='./cache/').to(device)
    
    # Load tokenizer for segments whose source language is Russian.
    tokenizer_rus = AutoTokenizer.from_pretrained(translator_model_name, token=True, src_lang="rus_Cyrl", cache_dir='./cache/')
    
    # Load tokenizer for segments whose source language is Spanish.
    tokenizer_spa = AutoTokenizer.from_pretrained(translator_model_name, token=True, src_lang="spa_Latn", cache_dir='./cache/')
    
    tokenizers = {
        'tokenizer_rus': tokenizer_rus,
        'tokenizer_spa': tokenizer_spa
    }
    
    # Translate train_val.
    childuid2translatedsegments_trainval = translate(
        childuid2segments_trainval, 
        childuid2doclang_trainval, 
        tokenizers, 
        model, 
        device
        )
    
    print("Writing translated segments (train_val) ...")
    with open(f'{output_dir}/childuid2translatedsegments_trainval.p', 'wb') as file:
        pickle.dump(childuid2translatedsegments_trainval, file)
    print("Done!")
    
    # Translate eval.
    childuid2translatedsegments_eval = translate(
        childuid2segments_eval, 
        childuid2doclang_eval, 
        tokenizers, 
        model, 
        device
        )

    print("Writing translated segments (eval) ...")
    with open(f'{output_dir}/childuid2translatedsegments_eval.p', 'wb') as file:
        pickle.dump(childuid2translatedsegments_eval, file)
    print("Done!")


if __name__ == "__main__":
    main()
    