"""
Claim Sentence Detection.
Step 2: Topic Classification using zero-shot NLI.
This is the python script version of claim_sentence_detection.ipynb and should be seen as the end version.

Model: BART-large  
Paper: https://aclanthology.org/2020.acl-main.703/  
HuggingFace: https://huggingface.co/facebook/bart-large-mnli

08.08.2023
"""

import requests
import json
from collections import defaultdict, Counter
from tqdm import tqdm
import dill as pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


PATH = '/mount/studenten/projekt-cs/crux2023/'
RAW_DATA_PATH = PATH + 'Datasets/'
PREPROCESSED_DATA_PATH = PATH + 'Project/DataPreprocessing/'
TOPIC_PATH = PREPROCESSED_DATA_PATH + 'Topics/outputs/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device, '\n')

# Load model and tokenizer.
nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli').to(device)
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')

# Load rootuidsegid2segment.
rootuidsegid2segment = pickle.load(open(PREPROCESSED_DATA_PATH + 'BuildMappings/outputs/rootuidsegid2segment.p', 'rb'))


def predict_claim_topic(premise:str, hypotheses:list[str], nli_model, tokenizer):
    """
    Zero-shot NLI for topic classification in claim sentence detection.
    
    Args:
        premise:str  The claim sentence to be classified.
        hypotheses:list[str]  The list of templates of the four pre-defined candidate topics.
        nli_model  The zero-shot NLI model for topic classification, here: BART-large.
        tokenizer  The tokenizer of the NLI model.
        
    Return:
        topic_pred:int  The index of the predicted topic in the hypotheses.
                        0:  'Characterizations of the Virus';
                        1:  'Contracting the virus';
                        2:  'Curing/Preventing/Destroying the Virus';
                        3:  'Government actions related to the virus';
                        4:  'Non-Pharmaceutical Interventions (NPIs): Masks';
                        5:  'Origin of the Virus';
                        6:  'Origin of the Virus: Virus Creation';
                        7:  'Transmitting the virus';
                        8:  'Treatment availability';
                        9:  'Treatment effectiveness';
                        10: 'Treatment safety'.
    
    """
    encoded_inputs = []
    for hypothesis in hypotheses:
        encoded_input = tokenizer.encode(premise, 
                                         hypothesis, 
                                         return_tensors='pt', 
                                         truncation_strategy='only_first')
        encoded_inputs.append(encoded_input)
    
    probabilities = []
    for enc_inp in encoded_inputs:
        logits = nli_model(enc_inp.to(device))[0]
        entail_contradiction_logits = logits[:, [0,2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        prob_label_is_true = probs[:, 1].cpu().detach().numpy()
        probabilities.append(prob_label_is_true)
    
    # print(probabilities)
    topic_pred = np.argmax(probabilities)
    # print(topic_pred)
    return topic_pred


def create_hypotheses(topics:list):
    """
    Contruct a hypothesis from each candidate topic.
    
    Args:
        topics:list  A list of candidate topics.
    
    Return:
        hypotheses:list  A list of hypotheses constructed from the candidate topics.
    """
    
    hypotheses = []
    for topic in topics:
        hypothesis = 'This is about {}.'.format(topic.lower())
        hypotheses.append(hypothesis)
    return hypotheses


# Topic Classification.
def predict_topics(input_path:str, output_path:str):
    """
    Predict the topic of each input claim sentence.
    
    Args:
       input_path:str   The input path.
       output_path:str  The output path.
        
    Return: None
    """
    
    # Load topics.
    topics = pickle.load(open(TOPIC_PATH + 'topics.p', 'rb'))
    
    # Create hypotheses.
    hypotheses = create_hypotheses(topics)
    
    # Load sentences that are predicted to contain claims by ClaimBuster.
    check_worthy_sents = pickle.load(open(input_path + 'check_worthy_sents.p', 'rb'))
    
    topic_predictions = defaultdict()
    for root_uid, segid in tqdm(check_worthy_sents):
        premise = rootuidsegid2segment[(root_uid, segid)]
        topic_predictions[(root_uid, segid)] = predict_claim_topic(premise, hypotheses, nli_model, tokenizer)

    topic_predictions = dict(topic_predictions)
    
    with open(output_path + 'rootuidsegid2topic.p', 'wb') as f:
        pickle.dump(topic_predictions, f)


def predict_subtopics(input_path:str, output_path:str):
    """
    Predict the subtopic of each input claim sentence.
    
    Args:
        input_path:str   The input path: /mount/studenten/projekt-cs/crux2023/Project/DataPreprocessing/Split/outputs/
        output_path:str  The output path: /mount/studenten/projekt-cs/crux2023/Project/DataPreprocessing/Split/outputs/
        ps: The input_path and output_path are identical in this case.
        
    Return: None
    """

    # Load subtopics.
    subtopics = pickle.load(open(TOPIC_PATH + 'subtopics.p', 'rb'))

    # Create hypotheses.
    hypotheses = create_hypotheses(subtopics)
    
    # Load sentences that are predicted to contain claims by ClaimBuster.
    check_worthy_sents = pickle.load(open(input_path + 'check_worthy_sents.p', 'rb'))

    subtopic_predictions = defaultdict()
    for root_uid, segid in tqdm(check_worthy_sents):
        premise = rootuidsegid2segment[(root_uid, segid)]
        subtopic_predictions[(root_uid, segid)] = predict_claim_topic(premise, hypotheses, nli_model, tokenizer)

    subtopic_predictions = dict(subtopic_predictions)
    
    with open(output_path + 'rootuidsegid2subtopic.p', 'wb') as f:
        pickle.dump(subtopic_predictions, f)


parser = ArgumentParser()
parser.add_argument('-i', '--input_dir', help='path to the input directory')
parser.add_argument('-o', '--output_dir', help='path to the output directory')
args = parser.parse_args()

print('Topic classification start...')
predict_topics(input_path=args.input_dir, output_path=args.output_dir)
print('Topic classification finished.')

print('Subtopic classification start...')
predict_subtopics(input_path=args.input_dir, output_path=args.output_dir)
print('Subtopic classification finished.')
