"""
Claim Sentence Detection.
Step 2: Topic Classification using zero-shot NLI.
This is the python script version of claim_sentence_detection.ipynb and should be seen as the end version.

Model: Llama2-Chat-hf
Paper: https://arxiv.org/abs/2307.09288
HuggingFace: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
"""

import os, torch
import xmltodict
import pandas as pd
from tqdm import tqdm
import dill as pickle

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


PATH = os.getenv('PATH')
DATA_PATH = os.getenv('DATA_PATH')
SOURCE_DATA_PATH = DATA_PATH + 'LDC/LDC2023E10_SMKBP_2023_Claim_Frame_Evaluation_Source_Data/'
CACHE_DIR = os.getenv('CACHE_DIR')

HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
device_name = torch.cuda.get_device_name()
print(f"Using device: {device} ({device_name})")

# Load model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', cache_dir=CACHE_DIR)
generative_nli_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', cache_dir=CACHE_DIR).to(device)

# Load the root_uids for the 250 evaluation source data.
with open(DATA_PATH + 'CRUX2023_Task1_Evaluation_root_uids.txt', 'r') as file:
    rootuids_eval = file.readlines()
    rootuids_eval = [rootuid.split('\n')[0] for rootuid in rootuids_eval]
    
# Create a mapping from parent_uid (i.e. root_uid) to child_uid.
parentuid2childuid = pd.read_csv(SOURCE_DATA_PATH + 'docs/parent_children.tab', sep='\t')
parentuid2childuid = parentuid2childuid[parentuid2childuid['child_asset_type']=='.ltf.xml']
parentuid2childuid = parentuid2childuid[['parent_uid', 'child_uid']]
parentuid2childuid = dict(zip(parentuid2childuid.parent_uid, parentuid2childuid.child_uid))

# Load check-worthy sentences, i.e. claim sentences, detected by ClaimBuster in check_worthiness_estimation.
claim_sents = pickle.load(open('./data/claim_sentences_eval.p', 'rb'))
claim_sents_eval = dict()
for pid in rootuids_eval:
    cid = parentuid2childuid[pid]
    claim_sents_eval[cid] = claim_sents[cid]
    
# Load the 3 topics, subtopic and claim templates for the evaluation.
topics = pd.read_csv(DATA_PATH + 'CRUX2023_Evaluation_Topics.tab.txt', sep='\t')

## 1. Topic ##
# Create a template for building prompts.
template_topic = f"""[CLAIMSENT] Which topic is this sentence about?
(A). {topics.iloc[0]['topic']}
(B). {topics.iloc[1]['topic']}
(C). {topics.iloc[2]['topic']}
(D). Non of above
Please choose one of the above options as the answer. Let's think step by step.
"""
print("template topic:\n", template_topic)

def create_prompt_topic(template, claim_sents):
    childuid2prompts = dict()
    for childuid in claim_sents_eval:
        segmentid2prompt = dict()
        for segmentid in claim_sents_eval[childuid]:
            claim_sent = claim_sents_eval[childuid][segmentid]
            prompt = template.replace('[CLAIMSENT]', claim_sent)
            segmentid2prompt[segmentid] = prompt
        childuid2prompts[childuid] = segmentid2prompt
    return childuid2prompts

childuid2prompts_topic = create_prompt_topic(template_topic, claim_sents_eval)


def predict_topic(inp, tokenizer, model):
    inputs = tokenizer(inp, return_tensors="pt").to(device)
    generate_ids = model.generate(inputs.input_ids, max_new_tokens=150)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output


# Topic Classification.
childuid2topic_outputs = dict()
for childuid, segmentid2prompt in tqdm(childuid2prompts_topic.items()):
    segmentid2topic = dict()
    for segmentid, prompt in segmentid2prompt.items():
        segmentid2topic[segmentid] = predict_topic(prompt, tokenizer, generative_nli_model)
    childuid2topic_outputs[childuid] = segmentid2topic

with open('./outputs/childuid2topic_outputs.p', 'wb') as f:
    pickle.dump(childuid2topic_outputs, f)
    
def extract_topic(childuid2topic):
    childuid2topics = dict()
    for childuid, segmentid2output in childuid2topic.items():
        segmentid2topic = dict()
        for segmentid, output in segmentid2output.items():
            output = output.split('the answer is ')[-1]
            if "(A)" in output:
                topic = "COVID-19 vaccine is harmful"
            elif "(B)" in output:
                topic = "Government actions related to the virus"
            elif "(C)" in output:
                topic = "Who is Sick / Who has tested positive"
            elif "(D)" in output:
                topic = "unk"
            else:
                topic = "unk"
            segmentid2topic[segmentid] = topic
        childuid2topics[childuid] = segmentid2topic
    return childuid2topics

childuid2topics = extract_topic(childuid2topic_outputs)

with open('./outputs/childuid2topics.p', 'wb') as file:
    pickle.dump(childuid2topics, file)
    
    
## 1. Subtopic ##
template_subtopic = f"""[CLAIMSENT]
This claim sentence is about the topic [TOPIC]. Which subtopic is it about?
(A). {topics.iloc[0]['subtopic']}
(B). {topics.iloc[1]['subtopic']}
(C). {topics.iloc[2]['subtopic']}
(D). Non of above
Please choose one of the above options as the answer. Let's think step by step.
"""
print("template subtopic:\n", template_subtopic)

def create_prompt_subtopic(template, claim_sents, childuid2topics):
    childuid2prompts = dict()
    for childuid in claim_sents_eval:
        segmentid2prompt = dict()
        for segmentid in claim_sents_eval[childuid]:
            claim_sent = claim_sents_eval[childuid][segmentid]
            prompt = template.replace('[CLAIMSENT]', claim_sent)
            
            if len(childuid2topics[childuid]) > 0 and childuid2topics[childuid][segmentid] != 'unk':
                topic = childuid2topics[childuid][segmentid]
                prompt = prompt.replace('[TOPIC]', topic)
            
            segmentid2prompt[segmentid] = prompt
        childuid2prompts[childuid] = segmentid2prompt
    return childuid2prompts

childuid2prompts_subtopic = create_prompt_subtopic(template_subtopic, claim_sents_eval, childuid2topics)

def predict_subtopic(inp, tokenizer, model):
    inputs = tokenizer(inp, return_tensors="pt").to(device)
    generate_ids = model.generate(inputs.input_ids, max_new_tokens=200)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output
    
childuid2subtopic_outputs = dict()
for childuid, segmentid2prompt in tqdm(childuid2prompts_subtopic.items()):
    segmentid2subtopic = dict()
    for segmentid, prompt in segmentid2prompt.items():
        segmentid2subtopic[segmentid] = predict_subtopic(prompt, tokenizer, generative_nli_model)
    childuid2subtopic_outputs[childuid] = segmentid2subtopic

with open('./outputs/childuid2subtopic_outputs.p', 'wb') as f:
    pickle.dump(childuid2subtopic_outputs, f)
    
def extract_subtopic(childuid2subtopic):
    childuid2subtopics = dict()
    for childuid, segmentid2output in childuid2subtopic.items():
        segmentid2subtopic = dict()
        for segmentid, output in segmentid2output.items():
            output = output.split('the answer is ')[-1]
            if "(A)" in output:
                subtopic = "COVID-19 vaccine causes medical conditions"
            elif "(B)" in output:
                subtopic = "Federal government seizing/diverting PPE and other medical supplies"
            elif "(C)" in output:
                subtopic = "Who has/had COVID-19"
            elif "(D)" in output:
                subtopic = "unk"
            else:
                subtopic = "unk"
            segmentid2subtopic[segmentid] = subtopic
        childuid2subtopics[childuid] = segmentid2subtopic
    return childuid2subtopics

childuid2subtopics = extract_subtopic(childuid2subtopic_outputs)

with open('./outputs/childuid2subtopics.p', 'wb') as file:
    pickle.dump(childuid2subtopics, file)

    
## 3. Claim Template ##
childuid2claimtemplate = dict()
for childuid in childuid2topics:
    segmentid2claimtemplate = dict()
    for segmentid in childuid2topics[childuid]:
        topic = childuid2topics[childuid][segmentid]
        claim_template = topics[topics['topic']==topic]['claim_template'].values.tolist()
        if len(claim_template) > 0:
            segmentid2claimtemplate[segmentid] = claim_template[0]
        else:
            segmentid2claimtemplate[segmentid] = 'unk'
    childuid2claimtemplate[childuid] = segmentid2claimtemplate
    
with open('./outputs/childuid2claimtemplate.p', 'wb') as file:
    pickle.dump(childuid2claimtemplate, file)
    
## Merge the outputs of topic, subtopic and claim template. ##
childuid2topic_subtopic_claimtemplate = dict()
for childuid, segmentid2topic in childuid2topics.items():
    segmentid2topic_subtopic_claimtemplate = dict()
    for segmentid, topic in segmentid2topic.items():
        subtopic = childuid2subtopics[childuid][segmentid]
        claim_template = childuid2claimtemplate[childuid][segmentid]
        segmentid2topic_subtopic_claimtemplate[segmentid] = [topic, subtopic, claim_template]
    childuid2topic_subtopic_claimtemplate[childuid] = segmentid2topic_subtopic_claimtemplate
    
with open('./outputs/childuid2topic_subtopic_claimtemplate.p', 'wb') as file:
    pickle.dump(childuid2topic_subtopic_claimtemplate, file)
    
print("Done!")
