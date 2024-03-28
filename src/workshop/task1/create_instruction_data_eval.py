import json, argparse
import pandas as pd
import dill as pickle
from tqdm import tqdm
from pathlib import Path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data_path', type=str, default='../../../datasets', help='Path to raw datasets.')
    parser.add_argument('-trp', '--trans_rsd_path', type=str, default='../../data_preprocessing/translate/output', help="Path to the translated rsd files.")
    parser.add_argument('--output_dir', type=str, default='./instruction_data', help='Path to the instruction data.')
    return parser.parse_args()

def construct_instruction_data(parent_children_df_eval, childuid2translatedrsd_eval, topic_list, subtopic_list):
    # Concatenate the topics in the topic list, each indexed by a number and separated by '\n'.
    topic_list_str = ""
    for i, topic in enumerate(topic_list):
        topic_list_str += f"{i+1}. {topic}\n"
    
    # Concatenate the subtopics in the subtopic list, each indexed by a number and separated by '\n'.
    subtopic_list_str = "\n".join(subtopic_list)
    for i, subtopic in enumerate(subtopic_list):
        subtopic_list_str += f"{i+1}. {subtopic}\n"

    # Construct CoT instruction data for the evaluation.
    cfe_eval = []

    for child_uid_rsd, translated_rsd in tqdm(childuid2translatedrsd_eval.items()):
        parent_uid = parent_children_df_eval[(parent_children_df_eval['child_uid']==child_uid_rsd) & 
                                             (parent_children_df_eval['child_asset_type']=='.ltf.xml')]['parent_uid'].values[0]

        question_event_relation_claim = f"Given the following list of candidate topics\n{topic_list_str}and list of subtopics\n{subtopic_list_str}. Which claims do the events and relations in this document comprise? Index your answers with numbers starting from 1. Let's think step by step.\n\nAnswers:\n[/INST]"
        
        text = {
            'child_uid': child_uid_rsd,
            'translated_rsd': translated_rsd
        }
        
        conversations = []

        prompt = {
            'speaker': 'User',
            'content': f"<s>[INST] <User> Given the following document:\n{translated_rsd}.\n\n" + question_event_relation_claim
        }
        conversations.append(prompt)

        cfe_eval.append(
            {
                'parent_uid': parent_uid,
                'text': text,
                'conversations': conversations
            }
        )
            
    return cfe_eval
            

def main():
    args = parse_arguments()
    dataset_path = Path(args.data_path)
    translated_rsd_path = Path(args.trans_rsd_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get the root_uids (i.e. parent_uids) that will be used for the evaluation.
    eval_root_uids_file = open(f"{dataset_path}/CRUX2023_Task1_Evaluation_root_uids.txt", "r")
    eval_root_uids = eval_root_uids_file.read().splitlines()

    # Load parent_children.tab.
    LDC2023E10_PATH = f"{dataset_path}/LDC/LDC2023E10_SMKBP_2023_Claim_Frame_Evaluation_Source_Data"
    parent_children_df = pd.read_csv(f"{LDC2023E10_PATH}/docs/parent_children.tab", sep='\t')

    # Get all ltf files in LDC2023E10 whose parent_uid is one of the 250 root_uids in CRUX2023_Task1_Evaluation_root_uids.txt.
    parent_children_df_eval = parent_children_df[(parent_children_df['parent_uid'].isin(eval_root_uids)) & (parent_children_df['child_asset_type'] == '.ltf.xml')]

    # Get the child_uids of the parent_uids that will be used for the evaluation.
    child_uids_eval = parent_children_df_eval['child_uid'].tolist()

    # Load all the 2011 translated rsd files in LDC2023E10_SMKBP_2023_Claim_Frame_Evaluation_Source_Data.
    childuid2translatedrsd = pickle.load(open(f'{translated_rsd_path}/childuid2translatedrsd_eval.p', "rb"))

    # Extract the 250 rsd files that will be used for the evaluation.
    childuid2translatedrsd_eval = {child_uid: childuid2translatedrsd[child_uid] for child_uid in child_uids_eval}
    
    # Write the mapping from child_uids to the extracted 250 rsd files.
    # with open(f"{output_dir}/childuid2translatedrsd_eval.p", 'wb') as file:
    #     pickle.dump(childuid2translatedrsd_eval, file)

    # Load the list of candidate topics for the evaluation.
    topics_df_eval = pd.read_csv(f"{dataset_path}/CRUX2023_Evaluation_Topics.tab.txt", sep='\t')

    topic_list = topics_df_eval['topic'].tolist()
    
    subtopic_list = topics_df_eval['subtopic'].tolist()
    
    cfe_eval = construct_instruction_data(parent_children_df_eval, childuid2translatedrsd_eval, topic_list, subtopic_list)
    
    with open(f"{output_dir}/instruction_data_eval.json", 'w') as file:
        json.dump(cfe_eval, file)


if __name__ == "__main__":
    main()
