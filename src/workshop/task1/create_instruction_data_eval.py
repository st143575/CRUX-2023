import argparse
import pandas as pd
import dill as pickle
from tqdm import tqdm
from pathlib import Path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='../../../datasets', help='Path to the raw datasets.')
    parser.add_argument('--output_dir', type=str, default='./instruction_data', help='Path to the instruction data.')
    return parser.parse_args()

def construct_instruction_data(childuid2translatedrsd_eval, topic_list, subtopic_list):
    # Concatenate the topics in the topic list, each indexed by a number and separated by '\n'.
    topic_list_str = ""
    for i, topic in enumerate(topic_list):
        topic_list_str += f"{i+1}. {topic}\n"
    
    # Concatenate the subtopics in the subtopic list, each indexed by a number and separated by '\n'.
    subtopic_list_str = "\n".join(subtopic_list)
    for i, subtopic in enumerate(subtopic_list):
        subtopic_list_str += f"{i+1}. {subtopic}\n"
    
    # Construct CoT instruction data for the evaluation.
    cfe_eval = dict()
    
    # parent_uids that are in childuid2translatedrsd (i.e. in the source data) but not in the ta3_ldc annotation.
    parent_uids_not_in_ta3_ldc = []

    for child_uid_rsd, translated_rsd in tqdm(eval_childuid2translatedrsd.items()):
        parent_uid = eval_parent_children_df[(eval_parent_children_df['child_uid']==child_uid_rsd) & 
                                             (eval_parent_children_df['child_asset_type']=='.ltf.xml')]['parent_uid'].values[0]
        print('parent_uid:', parent_uid, 'child_uid:', child_uid_rsd)

        if parent_uid not in list(claim_frames_df['root_uid']):
            parent_uids_not_in_ta3_ldc.append(parent_uid)
        else:
            question_event_relation_claim = f"Given the following list of candidate topics\n{topic_list_str}and list of subtopics\n{subtopic_list_str}. Which claims do the events and relations in this document comprise? Index your answers with numbers starting from 1. Let's think step by step.\n\nAnswers:\n"
            prompt = f"Given the following document:\n{translated_rsd}.\n\n" + question_event_relation_claim
            cfe_eval[parent_uid] = {child_uid: prompt}
            
    return cfe_eval, parent_uids_not_in_ta3_ldc
            

def main():
    args = parse_arguments()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get the root_uids (i.e. parent_uids) that will be used for the evaluation.
    eval_root_uids_file = open(f"{input_dir}/CRUX2023_Task1_Evaluation_root_uids.txt", "r")
    eval_root_uids = eval_root_uids_file.read().splitlines()
    print("Number of root_uids for evaluation:", len(eval_root_uids))

    # Load parent_children.tab.
    LDC2023E10_PATH = f"{input_dir}/LDC/LDC2023E10_SMKBP_2023_Claim_Frame_Evaluation_Source_Data"
    parent_children_df = pd.read_csv(f"{LDC2023E10_PATH}/docs/parent_children.tab", sep='\t')

    # Get all ltf files in LDC2023E10 whose parent_uid is one of the 250 root_uids in CRUX2023_Task1_Evaluation_root_uids.txt.
    parent_children_df_eval = parent_children_df[(parent_children_df['parent_uid'].isin(eval_root_uids)) & (parent_children_df['child_asset_type'] == '.ltf.xml')]

    # Get the child_uids of the parent_uids that will be used for the evaluation.
    child_uids_eval = parent_children_df_eval['child_uid'].tolist()

    # Load all the 2011 translated rsd files in LDC2023E10_SMKBP_2023_Claim_Frame_Evaluation_Source_Data.
    childuid2translatedrsd = pickle.load(open("../../data_preprocessing/translate/childuid2translatedrsd_eval.p", "rb"))

    # Extract the 250 rsd files that will be used for the evaluation.
    childuid2translatedrsd_eval = {child_uid: childuid2translatedrsd[child_uid] for child_uid in child_uids_eval}
    
    # Write the mapping from child_uids to the extracted 250 rsd files.
    with open(f"{output_dir}/childuid2translatedrsd_eval.p", 'wb') as file:
        pickle.dump(childuid2translatedrsd_eval, file)

    # Load the list of candidate topics for the evaluation.
    topics_df_eval = pd.read.csv(f"{input_dir}/CRUX2023_Evaluation_Topics.tab.txt", sep='\t')

    topic_list = topics_df_eval['topic'].tolist()
    
    subtopic_list = topics_df_eval['subtopic'].tolist()
    
    cfe_eval, parent_uids_not_in_ta3_ldc = construct_instruction_data(childuid2translatedrsd_eval, topic_list, subtopic_list)
    
    with open(f"{output_dir}/instruction_data_eval.json", 'w') as file:
        json.dump(cfe_eval, file)
    
    with open(f"{output_dir}/parent_uids_not_in_ta3_ldc_eval.txt", 'w') as file:
        file.write(parent_uids_not_in_ta3_ldc)


if __name__ == "__main__":
    main()
