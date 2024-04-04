import os, re, argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract structured claim frames from the model outputs and save them to a single file.')
    parser.add_argument('-i', '--input_dir', type=str, default='./eval_output', help="Path to the directory containing the model outputs")
    parser.add_argument('-o', '--output_dir', type=str, default='./claim_frames', help="Path to save the extracted claim frames")
    parser.add_argument('-d', '--data_dir', type=str, default='../../../datasets', help="Path to the datasets")
    return parser.parse_args()


def map_topic_subtopic_to_template(topic_df):
    topic_subtopic_to_template = {}
    for _, row in topic_df.iterrows():
        topic = row['topic']
        subtopic = row['subtopic']
        claim_template = row['claim_template']
        topic_subtopic_to_template[(topic, subtopic)] = claim_template
    return topic_subtopic_to_template


PATTERNS = {
    'topic': r"It's about the topic (.*?)\.", 
    'subtopic': r"It's about the subtopic (.*?)\.", 
    'x_variable': r"The object being claimed in the claim sentence is (.*?)\.", 
    'claimer': r"The claimer of this claim is (.*?)\.", 
    'epistemic_status': r"the epistemic status of the claimer to this claim is (.*?)\.",
    'sentiment_status': r"the claimer's sentiment status to this claim is (.*?)\.",
    'affiliation': r"The claimer of this claim is affiliated with (.*?)\.",
    'date_time': r"The claim was made at (.*?)\.", 
    'location': r"The claim was made at the location (.*?)\.", 
    'medium': r"the claim medium is (.*?)\.",
}

def extract_claim_component(claim, component_type, patterns):
    assert component_type in patterns
    pattern = patterns[component_type]
    match = re.search(pattern, claim)
    if match:
        extracted_string = match.group(1)
#         print(extracted_string, '\n')
        return extracted_string
    else:
        # print("No match found in the input string.\n")
        if component_type == 'topic':
            return 'None'
        if component_type == 'subtopic':
            return 'None'
        if component_type == 'x_variable':
            return 'None'
        if component_type == 'claimer':
            return 'author'
        if component_type == 'epistemic_status':
            return 'unknown'
        if component_type == 'sentiment_status':
            return 'neutral-unknown'
        if component_type == 'affilitation':
            return 'EMPTY_NA'
        if component_type == 'claim_datetime':
            return 'unknown'
        if component_type == 'location':
            return 'EMPTY_NA'
        if component_type == 'medium':
            return 'EMPTY_NA'

def get_claim_frames(input_dir, topic_subtopic2template, all_sentiment_status):
    for subdir, dirs, files in os.walk(input_dir):
        pid2claimframes = dict()
        for file in tqdm(files):
            if file.endswith('.txt'):
                fn = file.split('.')[0]
                parent_uid, child_uid = fn.split('_')[0], fn.split('_')[1]

                with open(f"{input_dir}/{file}", 'r') as f:
                    lines = f.readlines()
                    # print("LINES:\n", lines)

                    for i, line in enumerate(lines):
                        if line.startswith("Answers:\n"):
                            answers = lines[i+2:]
                            # print(len(answers), '\n', answers)
                            break
                    # print("ANSWERS:\n", len(answers))
                    # print('-'*50)

                    claim_frames = defaultdict(list)  # stores all claim frames from a file
                    for j, answer in enumerate(answers):
                        # print("ANSWER:\n", answer)
                        
                        # Extract the topic.
                        topic = extract_claim_component(answer, 'topic', PATTERNS)
                        # print(f"TOPIC: {topic}")

                        # Extract the subtopic.
                        subtopic = extract_claim_component(answer, 'subtopic', PATTERNS)
                        # print(f"SUBTOPIC: {subtopic}")

                        # Map topic and subtopic to claim template.
                        if (topic, subtopic) in topic_subtopic2template:
                            claim_template = topic_subtopic2template[(topic, subtopic)]
                        else:
                            claim_template = "No matched claim template found."
                        # print(f"CLAIM_TEMPLATE: {claim_template}")

                        # Extract the x_variable (i.e. claim object).
                        x_variable = extract_claim_component(answer, 'x_variable', PATTERNS)
                        # print(f"X_VARIABLE: {x_variable}")

                        # Extract the claimer.
                        claimer = extract_claim_component(answer, 'claimer', PATTERNS)
                        # print(f"CLAIMER: {claimer}")

                        # Extract the epistemic status (i.e. stance).
                        epistemic_status = extract_claim_component(answer, 'epistemic_status', PATTERNS)
                        # print(f"EPISTEMIC_STATUS: {epistemic_status}")

                        # Extract the affiliation.
                        affiliation = extract_claim_component(answer, 'affiliation', PATTERNS)
                        # print(f"AFFILIATION: {affiliation}")

                        # Extract the sentiment status.
                        sentiment_status = extract_claim_component(answer, 'sentiment_status', PATTERNS)
                        # Some of the model-generated answers to sentiment status are not correct, e.g. "neutral-unknownThe claimer of this claim is affiliated with EMPTY_NA".
                        # The candidate labels for sentiment status are ['positive', 'negative', 'mixed', 'neutral-unknown'].
                        # Therefore, we need to split the text span representing the label for the sentiment status from the rest of the string.
                        for sentiment_label in all_sentiment_status:
                            if sentiment_label in sentiment_status:
                                sentiment_status = sentiment_label
                                break
                        # print(f"SENTIMENT_STATUS: {sentiment_status}")

                        # Extract the date_time.
                        date_time = extract_claim_component(answer, 'date_time', PATTERNS)
                        # print(f"DATE_TIME: {date_time}")

                        # Extract the location.
                        location = extract_claim_component(answer, 'location', PATTERNS)
                        # print(f"LOCATION: {location}")

                        # Extract the medium.
                        medium = extract_claim_component(answer, 'medium', PATTERNS)
                        # print(f"MEDIUM: {medium}")

                        # Only store claim frames with valid topic, subtopic, and claim template.
                        if topic != 'None' and subtopic != 'None' and claim_template != 'No matched claim template found.':
                            claim_frame = {
                                'doc_id': parent_uid,
                                'claim_id': f"CL{parent_uid}.{str(j).zfill(6)}",
                                'topic': topic,
                                'subtopic': subtopic,
                                'claim_template': claim_template,
                                'x_variable': x_variable,
                                'claimer': claimer,
                                'epistemic_status': epistemic_status,
                                'affiliation': affiliation,
                                'sentiment_status': sentiment_status,
                                'date_time': date_time,
                                'location': location,
                                'medium': medium,
                            }
                            # print(f"Claim Frame {j}:\n", claim_frame)
                            # print('-'*50)
                            claim_frames[child_uid].append(claim_frame)
                    claim_frames = dict(claim_frames)
                pid2claimframes[parent_uid] = claim_frames
    return pid2claimframes

def main():
    args = parse_arguments()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    print("input_dir:", input_dir)
    print("output_dir:", output_dir)
    print("data_dir:", data_dir)

    # Load topic lists
    topic_df_trainval = pd.read_csv(f"{data_dir}/LDC/LDC2021E11_AIDA_Phase_3_Practice_Topic_Source_Data_V2.0/docs/topic_list.txt", sep='\t')
    topic_df_trainval = topic_df_trainval.rename(columns={'Template': 'claim_template'})
    topic_df_eval = pd.read_csv(f"{data_dir}/CRUX2023_Evaluation_Topics.tab.txt", sep='\t')
    topic_df = pd.concat([topic_df_trainval, topic_df_eval], ignore_index=True)
    
    topic_subtopic2template = map_topic_subtopic_to_template(topic_df)
    # print(f"Mapping from (topic, subtopic) to claim template:\n{topic_subtopic2template}\n")

    all_sentiment_status = ['positive', 'negative', 'mixed', 'neutral-unknown']

    # Extract claim frames from the model outputs.
    pid2claimframes = get_claim_frames(input_dir, topic_subtopic2template, all_sentiment_status)
    # print(len(pid2claimframes))
    
    # Save the extracted claim frames to a tab-separated file.
    claims_cleaned_tab = ""
    for parent_uid, child_uid2claimframes in pid2claimframes.items():
        for child_uid, claim_frames in child_uid2claimframes.items():
            for claim_frame in claim_frames:
                claims_cleaned_tab += f"{claim_frame['doc_id']}\t{claim_frame['claim_id']}\t{claim_frame['topic']}\t{claim_frame['claim_template']}\t{claim_frame['x_variable']}\t{claim_frame['claimer']}\t{claim_frame['epistemic_status']}\t{claim_frame['affiliation']}\t{claim_frame['sentiment_status']}\t{claim_frame['date_time']}\t{claim_frame['location']}\t{claim_frame['medium']}\n"

    with open(f"{output_dir}/claim_frames_task1_results.tab", "w") as f:
        f.write(claims_cleaned_tab)
        
    print(f"\nDone! Extracted claim frames are saved to ./{output_dir}/claim_frames_task1_results.tab.\n")


if __name__ == '__main__':
    main()