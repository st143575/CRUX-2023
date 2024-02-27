"""
Claim Sentence Detection.
Step 1: Check-worthiness Estimation using ClaimBuster.
This is the python script version of claim_sentence_detection.ipynb and should be seen as the end version.

ClaimBuster:
Paper: https://dl.acm.org/doi/abs/10.14778/3137765.3137815?casa_token=9JmDcizP4NcAAAAA%3AIgczZaYInxvWBE754WhgVdD-unWWCRgWJdv1vxcjVnWBu2Q6rIOGoGXJ1tGMLYflC7v25pDqsp1Ysw
API: https://idir.uta.edu/claimbuster/api/

08.08.2023
"""

import os
import requests
import json
from collections import defaultdict
from tqdm import tqdm
import dill as pickle
from argparse import ArgumentParser


PATH = '/mount/studenten/projekt-cs/crux2023/'
RAW_DATA_PATH = PATH + 'Datasets/'
PREPROCESSED_DATA_PATH = PATH + 'Project/DataPreprocessing/'

API_KEY = os.getenv('CLAIMBUSTER_API_KEY')


def run_claim_buster(sentence, api_key=API_KEY):
    """
    Run ClaimBuster for Claim-Worthiness-Estimation.
    
    Args:
        sentence:str  The input sentence.
        
    Return:
        api_response.json()  The JSON payload sent back by the ClaimBuster API.
    """
    
    # Define the endpoint (url) with the claim formatted as part of it, api-key (api-key is sent as an extra header)
    api_endpoint = f"https://idir.uta.edu/claimbuster/api/v2/score/text/{sentence}"
    request_headers = {"x-api-key": api_key}
    
    # Send the GET request to the API and store the api response
    api_response = requests.get(url=api_endpoint, headers=request_headers)
    
    # Return the JSON payload the API sent back
    return api_response.json()


def predict(input_path:str, output_path:str):
    # Load the test split.
    x_test_uids = pickle.load(open(input_path + 'x_test_uids.p, 'rb'))
    
    # Load rootuid2segments file.
    rootuid2segments = pickle.load(open(PREPROCESSED_DATA_PATH + 'BuildMappings/outputs/rootuid2segments.p', 'rb'))
    
    # check_worthy_sents is a dictionary that maps root_uid to {segid: segment}.
    check_worthy_sents = defaultdict()
    
    # Check worthiness estimation of examples in the test set.
    for root_uid in tqdm(x_test_uids):
        for segid, seg in rootuid2segments[root_uid].items():
            print(root_uid, segid)
            if root_uid == 'L0C04A2L' and segid == 'segment-3':
                score = -1
            elif root_uid == 'L0C049DQV' and segid == 'segment-11':
                score = -1
            else:
                output = run_claim_buster(seg)
                if output['results'] != {}:
                    score = output['results'][0]['score']
                else:
                    # For input sentences like '.', ClaimBuster will not return any results, but an emtpy dict {}.
                    # In this case, set score to -1.
                    score = -1
            # threshold = 0.7
            if score > 0.7:
                check_worthy_sents[(root_uid, segid)] = score
    check_worthy_sents = dict(check_worthy_sents)
    with open(output_path + 'check_worthy_sents.p', 'wb') as f:
        pickle.dump(check_worthy_sents, f)


parser = ArgumentParser()
parser.add_argument('-i', '--input_dir', help='path to the input directory')
parser.add_argument('-o', '--output_dir', help='path to the output directory')
args = parser.parse_args()

print('Claim sentence detection start...')
predict(input_path=args.input_dir, output_path=args.output_dir)
print('Claim sentence detection finished.')