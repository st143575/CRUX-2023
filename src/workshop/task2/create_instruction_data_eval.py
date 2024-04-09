import argparse, json, re
import dill as pickle
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create CoT instruction data.')
    parser.add_argument('-i', '--input_dir', type=str, default='../../../datasets', help='Path to raw datasets')
    parser.add_argument('-o', '--output_dir', type=str, default='./instruction_data', help='Path to save the instruction data')
    return parser.parse_args()


def get_rootuid_claimid_of_same_topic(eval_claim_frames, topic):
    rootuid_claimid_pairs = []
    for i, row_1 in eval_claim_frames.iterrows():
        for j, row_2 in eval_claim_frames.iterrows():
            if i != j:
                root_uid_1 = row_1['root_uid']
                claim_id_1 = row_1['claim_id']
                root_uid_2 = row_2['root_uid']
                claim_id_2 = row_2['claim_id']
                rootuid_claimid_pairs.append((root_uid_1, claim_id_1, root_uid_2, claim_id_2))
    return rootuid_claimid_pairs

def get_claim(root_uid, claim_id, claim_frames_eval):
    return claim_frames_eval[(root_uid, claim_id)]

def create_dialogue(ids, claim_frames_eval):
    """
    Create a prompt for a given claim pair
    (claim_id_1, root_uid_1, claim_id_2, root_uid_2): elements
    """
    dialogue = dict()
    claim_1 = get_claim(ids[0], ids[1], claim_frames_eval)
    claim_2 = get_claim(ids[2], ids[3], claim_frames_eval)
    prompt_q = f"""<s>{B_INST} {B_SYS}Two claims across two documents can have one of the following four relations:\n{REL_DEF}{E_SYS}{USER} Given the following two claims:\nClaim A: {str(claim_1)}, and\nClaim B: {str(claim_2)}.\nWhich of the following four relations do these two claims have? (a) identical, (b) refuted by, (c) supported by, or (d) related? Please only select one of these as the answer. Let's think step by step. {E_INST}
    """
    dialogue['User'] = prompt_q
    dialogue['Assistant'] = f"{ASSISTANT}"
    return dialogue


# Load special tokens.
with open("./special_tokens_task2.json", 'r') as file:
    special_tokens = json.load(file)
    
USER = special_tokens['USER']
ASSISTANT = special_tokens['ASSISTANT']
B_INST = special_tokens['B_INST']
E_INST = special_tokens['E_INST']
B_SYS = special_tokens['B_SYS']
E_SYS = special_tokens['E_SYS']
print("Special tokens:")
print(USER)
print(ASSISTANT)
print(B_INST)
print(E_INST)
print(B_SYS)
print(E_SYS)

# Load relation definitions.
with open("./relation_definitions.txt", 'r') as file:
    REL_DEF = file.read()
print("\nRelation definitions:\n", REL_DEF, '\n')


def main():
    args = parse_arguments()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print('Input directory:', input_dir)
    print('Output directory:', output_dir)

    # Load evaluation claim frames.
    claim_frames_eval_df = pd.read_csv(f'{input_dir}/CRUX2023_Task2_Evaluation_claim_frames.tab' , sep='\t')
    print("Number of evaluation claim frames:", len(claim_frames_eval_df))

    # Load evaluation topics.
    eval_topics_df = pd.read_csv(f'{input_dir}/CRUX2023_Evaluation_Topics.tab.txt', sep='\t')
    eval_topics = eval_topics_df['topic'].tolist()
    print("Evaluation topics:", eval_topics)

    # Topic 1: "COVID-19 vaccine is harmful"
    eval_claim_frames_topic1_df = claim_frames_eval_df[claim_frames_eval_df['topic'] == eval_topics[0]]

    # Topic 2: "Government actions related to the virus"
    eval_claim_frames_topic2_df = claim_frames_eval_df[claim_frames_eval_df['topic'] == eval_topics[1]]

    # Topic 3: "Who is Sick / Who has tested positive"
    eval_claim_frames_topic3_df = claim_frames_eval_df[claim_frames_eval_df['topic'] == eval_topics[2]]

    rootuid_claimid_pairs_topic1 = get_rootuid_claimid_of_same_topic(
        eval_claim_frames_topic1_df, 
        eval_topics[0]
    )
    print("Length of rootuid_claimid_topic1:", len(rootuid_claimid_pairs_topic1))

    rootuid_claimid_pairs_topic2 = get_rootuid_claimid_of_same_topic(
        eval_claim_frames_topic2_df, 
        eval_topics[1]
    )
    print("Length of rootuid_claimid_topic2:", len(rootuid_claimid_pairs_topic2))

    rootuid_claimid_pairs_topic3 = get_rootuid_claimid_of_same_topic(
        eval_claim_frames_topic3_df, 
        eval_topics[2]
    )
    print("Length of rootuid_claimid_topic3:", len(rootuid_claimid_pairs_topic3))

    topic2rootuidclaimidpairs = {
        eval_topics[0]: rootuid_claimid_pairs_topic1,
        eval_topics[1]: rootuid_claimid_pairs_topic2,
        eval_topics[2]: rootuid_claimid_pairs_topic3,
    }

    with open(f'{output_dir}/topic2rootuidclaimidpairs.p', 'wb') as file:
        pickle.dump(topic2rootuidclaimidpairs, file)

    claim_frames_eval = claim_frames_eval_df.to_dict('records')
    claim_frames_eval = {(claim['root_uid'], claim['claim_id']): claim for claim in claim_frames_eval}
    print("Length of claim_frames_eval:", len(claim_frames_eval))

    # Create instruction data for all claim pairs.
    instruction_data = dict()
    for topic, rootuid_claimid_pairs in topic2rootuidclaimidpairs.items():
        print(f"\nCreating instruction data for topic: {topic}")
        for ids in tqdm(rootuid_claimid_pairs):
            dialogue = create_dialogue(ids, claim_frames_eval)
            instruction_data[ids] = dialogue
    print("Number of instruction data:", len(instruction_data))
    print("Example instruction data:\n", instruction_data[topic2rootuidclaimidpairs[eval_topics[0]][0]])

    # Save instruction data.
    with open(f'{output_dir}/instruction_data_eval.p', 'wb') as file:
        pickle.dump(instruction_data, file)
    print(f"Instruction data for evaluating on Task 2 has been saved at {output_dir}/instruction_data_eval.p")


if __name__ == '__main__':
    main()