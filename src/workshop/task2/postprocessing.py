import os, re, argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract structured claim frames from the model outputs and save them to a single file.')
    parser.add_argument('-i', '--input_dir', type=str, default='./eval_output', help="Path to the directory containing the model outputs")
    parser.add_argument('-o', '--output_dir', type=str, default='./cross_claim_relations', help="Path to save the extracted claim frames")
    return parser.parse_args()

def extract_answer(output):
    pattern = r"\(([abcd])\) (\w+(?: \w+)?)\."
    match = re.search(pattern, output)
    if match:
        return match.group(2)
    else:
        return None

def get_ids_from_filename(filename):
    ids = filename.split('/')[1]
    ids = ids.split('.txt')[0]
    ids = ids.split('_')
    # print("IDs:", ids)
    doc_id_1 = ids[0]
    claim_id_1 = ids[1]
    doc_id_2 = ids[2]
    claim_id_2 = ids[3]
    return doc_id_1, claim_id_1, doc_id_2, claim_id_2


def extract_relations_from_files(input_dir):
    """
    Extract cross-claim relations from the model output files.
    """
    ids2relation = dict()
    for subdir, dirs, files in os.walk(input_dir):
        for file in tqdm(files):
            if file.endswith('.txt'):
                with open(os.path.join(subdir, file), 'r') as file:
                    doc_id_1, claim_id_1, doc_id_2, claim_id_2 = get_ids_from_filename(file.name)
                    lines = file.readlines()
                    answer_span = lines[-1].strip()
                    answer = extract_answer(answer_span)
                    # In the final output, the labels 'supported by' and 'refuted by' should be 
                    # 'supported_by' and 'refuted_by', respectively.
                    # Convert 'supported by' to 'supported_by' and 'refuted by' to 'refuted_by'.
                    if answer == 'supported by':
                        answer = 'supported_by'
                    if answer == 'refuted by':
                        answer = 'refuted_by'
                    if answer is not None:
                        ids2relation[(doc_id_1, claim_id_1, doc_id_2, claim_id_2)] = answer
    return ids2relation

def write_output(output_dir, ids2relation):
    with open(output_dir / 'cross_claim_relations.tab', 'w') as file:
        for (doc_id_1, claim_id_1, doc_id_2, claim_id_2), relation in ids2relation.items():
            file.write(f"{claim_id_1}\t{relation}\t{claim_id_2}\n")


def main():
    args = parse_arguments()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Input directory:", input_dir)
    print("Output directory:", output_dir)

    # Extract cross-claim relations from the model output files.
    ids2relation = extract_relations_from_files(input_dir)
                
    # Write the extracted cross-claim relations to a tab-separate file.
    write_output(output_dir, ids2relation)
    print(f"Cross-claim relations extracted and saved to {output_dir}/cross_claim_relations_task2_results.tab")

if __name__ == '__main__':
    main()