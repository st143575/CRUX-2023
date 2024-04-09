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

def get_topic(claim_frames_df, root_uid, claim_id):
    return claim_frames_df[(claim_frames_df['root_uid'] == root_uid) & (claim_frames_df['claim_id']==claim_id)]['topic'].values[0]

def get_subtopic(claim_frames_df, root_uid, claim_id):
    return claim_frames_df[(claim_frames_df['root_uid'] == root_uid) & (claim_frames_df['claim_id']==claim_id)]['subtopic'].values[0]

def get_claim_template(claim_frames_df, root_uid, claim_id):
    return claim_frames_df[(claim_frames_df['root_uid'] == root_uid) & (claim_frames_df['claim_id']==claim_id)]['claim_template'].values[0]

def get_x_variable(claim_frames_df, root_uid, claim_id):
    return claim_frames_df[(claim_frames_df['root_uid'] == root_uid) & (claim_frames_df['claim_id']==claim_id)]['x_variable'].values[0]

def get_x_variable_identity(claim_frames_df, root_uid, claim_id):
    return claim_frames_df[(claim_frames_df['root_uid'] == root_uid) & (claim_frames_df['claim_id']==claim_id)]['qnode_x_variable_identity'].values[0]

def get_description(claim_frames_df, root_uid, claim_id):
    return claim_frames_df[(claim_frames_df['root_uid'] == root_uid) & (claim_frames_df['claim_id']==claim_id)]['description'].values[0]

def get_claimer(claim_frames_df, root_uid, claim_id):
    return claim_frames_df[(claim_frames_df['root_uid'] == root_uid) & (claim_frames_df['claim_id']==claim_id)]['claimer'].values[0]

def get_claimer_identity(claim_frames_df, root_uid, claim_id):
    return claim_frames_df[(claim_frames_df['root_uid'] == root_uid) & (claim_frames_df['claim_id']==claim_id)]['qnode_claimer_identity'].values[0]

def get_affiliation(claim_frames_df, root_uid, claim_id):
    return claim_frames_df[(claim_frames_df['root_uid'] == root_uid) & (claim_frames_df['claim_id']==claim_id)]['claimer_affiliation'].values[0]

def get_epistemic_status(claim_frames_df, root_uid, claim_id, mode='status'):
    epistemic_status = claim_frames_df[(claim_frames_df['root_uid'] == root_uid) & (claim_frames_df['claim_id']==claim_id)]['epistemic_status'].values[0]
    if mode == 'status':
        return epistemic_status
    elif mode == 'truth':
        epistemic_truth = epistemic_status if epistemic_status == "unknown" else epistemic_status.split('-')[0]
        return epistemic_truth
    elif mode == 'certainty':
        epistemic_certainty = epistemic_status if epistemic_status == "unknown" else epistemic_status.split('-')[1]
        return epistemic_certainty
    else:
        raise ValueError(f"Invalid mode: {mode}! Must be one of ['status', 'truth', 'certainty'].")
    
def get_sentiment(claim_frames_df, root_uid, claim_id):
    return claim_frames_df[(claim_frames_df['root_uid'] == root_uid) & (claim_frames_df['claim_id']==claim_id)]['sentiment_status'].values[0]

def get_datetime(claim_frames_df, root_uid, claim_id):
    return claim_frames_df[(claim_frames_df['root_uid'] == root_uid) & (claim_frames_df['claim_id']==claim_id)]['claim_datetime'].values[0]

def get_location(claim_frames_df, root_uid, claim_id):
    return claim_frames_df[(claim_frames_df['root_uid'] == root_uid) & (claim_frames_df['claim_id']==claim_id)]['claim_location'].values[0]

def get_medium(claim_frames_df, root_uid, claim_id):
    return claim_frames_df[(claim_frames_df['root_uid'] == root_uid) & (claim_frames_df['claim_id']==claim_id)]['claim_medium'].values[0]

def map_claimids_rootuids_to_elements(cross_claim_relations_df, claim_frames_df):
    claimids_rootuids2elements = dict()
    for _, row in tqdm(cross_claim_relations_df.iterrows()):
        ### Claim 1 ###
        # claim_id
        claim_id_1 = row['claim_id_1']
        # root_uid (parent_uid)
        root_uid_1 = row['root_uid_1']
        # topic
        topic_1 = get_topic(claim_frames_df, root_uid_1, claim_id_1)
        # subtopic
        subtopic_1 = get_subtopic(claim_frames_df, root_uid_1, claim_id_1)
        # claim_template
        claim_template_1 = get_claim_template(claim_frames_df, root_uid_1, claim_id_1)
        # x_variable_identity_qnode
        qnode_x_variable_identity_1 = get_x_variable_identity(claim_frames_df, root_uid_1, claim_id_1)
        # x_variable
        x_variable_1 = get_x_variable(claim_frames_df, root_uid_1, claim_id_1)
        # claimer_identity_qnode
        qnode_claimer_identity_1 = get_claimer_identity(claim_frames_df, root_uid_1, claim_id_1)
        # claimer
        claimer_1 = get_claimer(claim_frames_df, root_uid_1, claim_id_1)
        # epistemic_truth
        epistemic_truth_1 = get_epistemic_status(claim_frames_df, root_uid_1, claim_id_1, mode='truth')

        ### Claim 2 ###
        # claim_id
        claim_id_2 = row['claim_id_2']
        # root_uid (parent_uid)
        root_uid_2 = row['root_uid_2']
        # topic
        topic_2 = get_topic(claim_frames_df, root_uid_2, claim_id_2)
        # subtopic
        subtopic_2 = get_subtopic(claim_frames_df, root_uid_2, claim_id_2)
        # claim_template
        claim_template_2 = get_claim_template(claim_frames_df, root_uid_2, claim_id_2)
        # x_variable_identity_qnode
        qnode_x_variable_identity_2 = get_x_variable_identity(claim_frames_df, root_uid_2, claim_id_2)
        # x_variable
        x_variable_2 = get_x_variable(claim_frames_df, root_uid_2, claim_id_2)
        # claimer_identity_qnode
        qnode_claimer_identity_2 = get_claimer_identity(claim_frames_df, root_uid_2, claim_id_2)
        # claimer
        claimer_2 = get_claimer(claim_frames_df, root_uid_2, claim_id_2)
        # epistemic_truth
        epistemic_truth_2 = get_epistemic_status(claim_frames_df, root_uid_2, claim_id_2, mode='truth')

        claimids_rootuids2elements[(claim_id_1, root_uid_1, claim_id_2, root_uid_2)] = {
            'topic_1': topic_1,
            'subtopic_1': subtopic_1,
            'claim_template_1': claim_template_1,
            'qnode_x_variable_identity_1': qnode_x_variable_identity_1,
            'x_variable_1': x_variable_1,
            'qnode_claimer_identity_1': qnode_claimer_identity_1,
            'claimer_1': claimer_1,
            'epistemic_truth_1': epistemic_truth_1,
            'topic_2': topic_2,
            'subtopic_2': subtopic_2,
            'claim_template_2': claim_template_2,
            'qnode_x_variable_identity_2': qnode_x_variable_identity_2,
            'x_variable_2': x_variable_2,
            'qnode_claimer_identity_2': qnode_claimer_identity_2,
            'claimer_2': claimer_2,
            'epistemic_truth_2': epistemic_truth_2,
            'relation': row['relation']
        }
    return claimids_rootuids2elements

def get_claim(claim_id, root_uid, claim_frames_df):
    claim = {
        # 'doc_id': root_uid,
        # 'claim_id': claim_id,
        # 'description': get_description(claim_frames_df, root_uid, claim_id),
        'topic': get_topic(claim_frames_df, root_uid, claim_id),
        'subtopic': get_subtopic(claim_frames_df, root_uid, claim_id),
        'claim_template': get_claim_template(claim_frames_df, root_uid, claim_id),
        'x_variable': get_x_variable(claim_frames_df, root_uid, claim_id),
        'claimer': get_claimer(claim_frames_df, root_uid, claim_id),
        'epistemic_status': get_epistemic_status(claim_frames_df, root_uid, claim_id),
        'affiliation': get_affiliation(claim_frames_df, root_uid, claim_id),
        'sentiment': get_sentiment(claim_frames_df, root_uid, claim_id),
        'datetime': get_datetime(claim_frames_df, root_uid, claim_id),
        'location': get_location(claim_frames_df, root_uid, claim_id),
        'medium': get_medium(claim_frames_df, root_uid, claim_id)
    }
    return claim

def populate_claim_template(claim_template, x_variable):
    core_claim = re.sub(r'(\w*/)*\w*-?[Xx]', x_variable, claim_template)
    return core_claim


def same_topic(elements) -> bool:
    return elements['topic_1'] == elements['topic_2']

def same_subtopic(elements) -> bool:
    return elements['subtopic_1'] == elements['subtopic_2']

def same_claim_template(elements) -> bool:
    return elements['claim_template_1'] == elements['claim_template_2']

def same_x_variable_identity(elements) -> bool:
    # Compare qnode_x_variable_identity rather than x_variable, since different x_variables can be mapped to the same qnode.
    return elements['qnode_x_variable_identity_1'] == elements['qnode_x_variable_identity_2']

def same_claimer_identity(elements) -> bool:
    # Compare qnode_x_claimer_identity rather than claimer.
    return elements['qnode_claimer_identity_1'] == elements['qnode_claimer_identity_2']

def same_epistemic_truth(elements) -> bool:
    return elements['epistemic_truth_1'] == elements['epistemic_truth_2']

def is_identical(elements):
    return same_topic(elements) and same_subtopic(elements) and same_claim_template(elements) and same_x_variable_identity(elements) and same_epistemic_truth(elements)





def create_dialogue(ids, elements, claim_frames_df):
    """
    Create a prompt for a given claim pair
    (claim_id_1, root_uid_1, claim_id_2, root_uid_2): elements
    """
    dialogue = dict()
    claim_1 = get_claim(ids[0], ids[1], claim_frames_df)
    claim_2 = get_claim(ids[2], ids[3], claim_frames_df)
    core_claim_1 = populate_claim_template(claim_1['claim_template'], claim_1['x_variable'])
    core_claim_2 = populate_claim_template(claim_2['claim_template'], claim_2['x_variable'])
    prompt_q = f"""<s>{B_INST} {B_SYS}Two claims across two documents can have one of the following four relations:\n{REL_DEF}{E_SYS}{USER} Given the following two claims:\nClaim A: {str(claim_1)}, and\nClaim B: {str(claim_2)}.\nWhich of the following four relations do these two claims have? (a) identical, (b) refuted by, (c) supported by, or (d) related? Please only select one of these as the answer. Let's think step by step. {E_INST}
    """

    prompt_a = f"""{ASSISTANT} The first claim is Claim A: {str(claim_1)}. The second claim is Claim B: {str(claim_2)}. First, let's check if Claim A and Claim B are identical. """

    topic_a = f"Claim A has the topic {elements['topic_1']} and Claim B has the topic {elements['topic_2']}. "
    subtopic_a = f"Claim A has the subtopic {elements['subtopic_1']} and Claim B has the subtopic {elements['subtopic_2']}. "
    claim_template_a = f"Claim A has the claim template {elements['claim_template_1']} and Claim B has the claim template {elements['claim_template_2']}. "
    x_variable_identity_a = f"Claim A has the X variable {elements['x_variable_1']} and Claim B has the X variable {elements['x_variable_2']}. "
    claimer_identity_a = f"Claim A is made by the claimer {elements['claimer_1']} and Claim B is made by the claimer {elements['claimer_2']}. "
    epistemic_truth_a = ""

    ### Identical ###
    # If two claims are identical, just declare that they have the same elements. 
    # For epistemic truth, extract the epistemic truth values of the two claims from the annotation and declare that they are the same.
    if elements['relation'] == 'identical':
        topic_a += "Thus, Claim A and Claim B have the same topic. "
        subtopic_a += "Thus, Claim A and Claim B have the same subtopic. "
        claim_template_a += "Thus, Claim A and Claim B have the same claim template. "
        x_variable_identity_a += "Thus, Claim A and Claim B have the same X variable identity. "
        claimer_identity_a += "Thus, Claim A and Claim B are made by the same claimer identity. "
        epistemic_truth_a += f"The epistemic truth value of the Claim A is {elements['epistemic_truth_1']} and the epistemic truth value of Claim B is {elements['epistemic_truth_2']}. Thus, Claim A and Claim B have the same epistemic truth value. "
        prompt_a += "".join([topic_a, subtopic_a, claim_template_a, x_variable_identity_a, claimer_identity_a, epistemic_truth_a])
        prompt_a += f"Therefore, Claim A and Claim B are {elements['relation']}. The answer is: (a) identical.</s> "

    ### Refute ###
    # If Claim A is refuted by Claim B, check which rule(s) is/are broken.
    elif elements['relation'] == 'refuted_by':
        # If refute, then the two claims must have the same topic.
        topic_a += "Thus, Claim A and Claim B have the same topic. "
            
        # If refute, then the two claims should not be identical.
        # There are multiple possible rule-breaks that can make two claims not identical.
        # (1) different subtopics
        if not same_subtopic(elements):
            subtopic_a += "Thus, Claim A and Claim B have different subtopics. This makes them not identical. "
        else:
            subtopic_a += "Thus, Claim A and Claim B have the same subtopic. "
        
        # (2) different claim templates
        if not same_claim_template(elements):
            claim_template_a += "Thus, Claim A and Claim B have different claim templates. This makes them not identical. "
        else:
            claim_template_a += "Thus, Claim A and Claim B have the same claim template. "
            
        # (3) different x variable identities
        if not same_x_variable_identity(elements):
            x_variable_identity_a += "Thus, Claim A and Claim B have different X variable identities. This makes them not identical. "
        else:
            x_variable_identity_a += "Thus, Claim A and Claim B have the same X variable identity. "
        
        # (4) different claimer identities
        if not same_claimer_identity(elements):
            claimer_identity_a += "Thus, Claim A and Claim B have different claimer identities. This makes them not identical. "
        else:
            claimer_identity_a += "Thus, Claim A and Claim B have the same claimer identity. "
            
        # If Claim A is true, then Claim B cannot be true.
        claimer_identity_a += f"If {core_claim_1} is true, then it cannot be true that {core_claim_2}. "
        
        prompt_a += "".join([topic_a, subtopic_a, claim_template_a, x_variable_identity_a, claimer_identity_a, epistemic_truth_a])
        prompt_a += f"Therefore, Claim A is refuted by Claim B. The answer is (b) refuted by.</s> " 

    ### Support ###
    elif elements['relation'] == 'supported_by':
        # If support, then the two claims must have the same topic.
        prompt_a += f"Thus, Claim A and Claim B have the same topic. "
        
        # If support, then the two claims should not be identical.
        # There are multiple possible rule-breaks that can make two claims not identical.
        # (1) different subtopics
        if not same_subtopic(elements):
            subtopic_a += "Thus, Claim A and Claim B have different subtopics. This makes them not identical. "
        else:
            subtopic_a += "Thus, Claim A and Claim B have the same subtopic. "
        
        # (2) different claim templates
        if not same_claim_template(elements):
            claim_template_a += "Thus, Claim A and Claim B have different claim templates. This makes them not identical. "
        else:
            claim_template_a += "Thus, Claim A and Claim B have the same claim template. "
            
        # (3) different x variable identities
        if not same_x_variable_identity(elements):
            x_variable_identity_a += "Thus, Claim A and Claim B have different X variable identity qnodes. This makes them not identical. "
        else:
            x_variable_identity_a += "Thus, Claim A and Claim B have the same X variable identity qnode. "
        
        # (4) different claimer identities
        if not same_claimer_identity(elements):
            claimer_identity_a += f"Thus, Claim A and Claim B have different claimer identities. This makes them not identical. "
        else:
            claimer_identity_a += "Thus, Claim A and Claim B have the same claimer identity. "
            
        # (5) If Claim A is true, then Claim B can still be true.
        claimer_identity_a += f"If {core_claim_1} is true, then {core_claim_2} can still be true. This makes the first claim not refuted by the second claim. "
            
        # If support, then the two claims should not be refuting.
        # If Claim A is true, then Claim B is more plausible.
        claimer_identity_a += f"If {core_claim_1} is true, then it is more plausible that {core_claim_2}. "
        
        prompt_a += "".join([topic_a, subtopic_a, claim_template_a, x_variable_identity_a, claimer_identity_a, epistemic_truth_a])
        prompt_a += f"Therefore, Claim A is supported by Claim B. The answer is (c) supported by.</s> "

    ### Related ###
    elif elements['relation'] == 'related':
        # If related, then the two claims must have the same topic.
        prompt_a += f"Thus, Claim A and Claim B have the same topic. "
        
        # If related, then the two claims should not be identical.
        # There are multiple possible rule-breaks that can make two claims not identical.
        # (1) different subtopics
        if not same_subtopic(elements):
            subtopic_a += "Thus, Claim A and Claim B have different subtopics. This makes them not identical. "
        else:
            subtopic_a += "Thus, Claim A and Claim B have the same subtopic. "
        
        # (2) different claim templates
        if not same_claim_template(elements):
            claim_template_a += "Thus, Claim A and Claim B have different claim templates. This makes them not identical. "
        else:
            claim_template_a += "Thus, Claim A and Claim B have the same claim template. "
            
        # (3) different x variable identities
        if not same_x_variable_identity(elements):
            x_variable_identity_a += "Thus, Claim A and Claim B have different X variable identity qnodes. This makes them not identical. "
        else:
            x_variable_identity_a += "Thus, Claim A and Claim B have the same X variable identity qnode. "
        
        # (4) different claimer identities
        if not same_claimer_identity(elements):
            claimer_identity_a += "Thus, Claim A and Claim B have different claimer identities. This makes them not identical. "
        else:
            claimer_identity_a += "Thus, Claim A and Claim B have the same claimer identity. "
        
        # If related, then the two claims should not be refuting.
        # If Claim A is true, then Claim B can still be true.
        claimer_identity_a += f"If {core_claim_1} is true, then {core_claim_2} can still be true. This makes Claim A not refuted by Claim B. "
        
        # If related, then the two claims should not be supporting.
        # There are multiple possible rule-breaks that can make two claims not supporting.
        # (1) They cannot both be true at the same time.
        claimer_identity_a += f"Claim A and Claim B cannot both be true at the same time. "
        
        # (2) If Claim A is true, it does not imply that Claim B is more plausible.
        claimer_identity_a += f"If Claim A is true, it does not imply that Claim B is more plausible. "
        
        # If Claim A is true, it doesn't affect the plausibility of Claim B.
        claimer_identity_a += f"Instead, it does not affect the plausibility of the second claim. "
        
        prompt_a += "".join([topic_a, subtopic_a, claim_template_a, x_variable_identity_a, claimer_identity_a, epistemic_truth_a])
        prompt_a += f"Therefore, Claim A is related to Claim B. The answer is (d) related.</s> "

    dialogue['User'] = prompt_q
    dialogue['Assistant'] = prompt_a
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

    ANNOTATION_PATH = f"{input_dir}/LDC/LDC2021E16_AIDA_Phase_3_TA3_Practice_Topic_Annotation_V5.1"

    # Load cross claim relations.
    ccr_df = pd.read_csv(f"{ANNOTATION_PATH}/data/ta3_ldc/cross_claim_relations.tab", sep='\t')

    # Load claim frames.
    claim_frames_df = pd.read_csv(f"{ANNOTATION_PATH}/data/ta3_ldc/claim_frames.tab", sep='\t')
    claim_templates = list(set(claim_frames_df['claim_template']))

    # Get claimids_rootuids2elements.
    claimids_rootuids2elements = map_claimids_rootuids_to_elements(ccr_df, claim_frames_df)

    # example = list(claimids_rootuids2elements.items())[0]
    # example_prompt = create_dialogue(example[0], example[1], claim_frames_df)
    # print(example_prompt)

    # Create instruction data for all claim pairs.
    instruction_data = dict()
    for ids, elements in tqdm(claimids_rootuids2elements.items()):
        instruction_data[ids] = create_dialogue(ids, elements, claim_frames_df)
    print("Number of instruction data:", len(instruction_data))
    print("Example instruction data:\n", instruction_data[list(instruction_data.keys())[0]])

    # Save instruction data.
    with open(f"{output_dir}/instruction_data_ft.p", 'wb') as file:
        pickle.dump(instruction_data, file)
    print(f"Instruction data for fine-tuning on Task 2 has been saved at {output_dir}/instruction_data_ft.p")


if __name__ == '__main__':
    main()