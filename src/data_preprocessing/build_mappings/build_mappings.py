import os, argparse, json, xmltodict
import pandas as pd
import numpy as np
import dill as pickle
from collections import Counter, defaultdict
from tqdm import tqdm
from pathlib import Path


def parse_arguments():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(description='Build mappings between datasets.')
    parser.add_argument('-i', '--input_dir', type=str, default='../../datasets', help="Path to the datasets")
    parser.add_argument('-o', '--output_dir', type=str, default='./output', help="Path to save the mappings")
    return parser.parse_args()


def main():
    args = parse_arguments()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Input directory:", input_dir)
    print("Output directory:", output_dir)
    
    LDC2021E11_PATH = f'{input_dir}/LDC/LDC2021E11_AIDA_Phase_3_Practice_Topic_Source_Data_V2.0/'
    LDC2021E16_PATH = f'{input_dir}/LDC/LDC2021E16_AIDA_Phase_3_TA3_Practice_Topic_Annotation_V5.1/'
    DWD_PATH = f'{input_dir}/DWD_Overlay/'
    
    parent_children_df = pd.read_csv(LDC2021E11_PATH + 'docs/parent_children.tab', sep='\t')
    
    # Build a mapping from parent_uid (i.e. root_uid) to children_uid, regardless of the type of the file.
    parentuid2childuid2assettype = defaultdict()
    for idx, row in parent_children_df.iterrows():
        childuid2assettype = {str(row['child_uid']): row['child_asset_type']}
        parentuid2childuid2assettype[row['parent_uid']] = childuid2assettype
    parentuid2childuid2assettype = dict(parentuid2childuid2assettype)
    
    # Write the results to a parentuid2childuid.p.
    with open(f"{output_dir}/LDC/parentuid2childuid2assettype.p", 'wb') as file:
        pickle.dump(parentuid2childuid2assettype, file)
              
    
    # Build a mapping from parent_uid to child_uid of the ltf file belonging to the HTML document represented by that parent_uid.
    parentuid2ltfchilduid = defaultdict()
    for idx, row in parent_children_df.iterrows():
        if row['child_asset_type'] == '.ltf.xml':
            parentuid2ltfchilduid[row['parent_uid']] = str(row['child_uid'])
    parentuid2ltfchilduid = dict(parentuid2ltfchilduid)
    
    # Write the results to a file
    with open(f'{output_dir}/LDC/parentuid2ltfchilduid.p', 'wb') as file:
        pickle.dump(parentuid2ltfchilduid, file)
        
        
    # Build a mapping from (child_uid, child_asset_type) to parent_uid.
    childuid_assettype2parentuid = defaultdict(list)
    for idx, row in parent_children_df.iterrows():
        childuid_assettype2parentuid[(str(row['child_uid']), row['child_asset_type'])].append(row['parent_uid'])
    childuid_assettype2parentuid = dict(childuid_assettype2parentuid)
    
    # Write the results to childuid_assettype2parentuid.p.
    with open(f'{output_dir}/LDC/childuid_&_assettype2parentuid.p', 'wb') as file:
        pickle.dump(childuid_assettype2parentuid, file)
    
    
    # Create a mapping from child_uid (equivalent to the file name) to the content of the rsd file, i.e. 
    # the raw text of the article.
    LTF_PATH = f"{input_dir}/LDC/LDC2021E11_AIDA_Phase_3_Practice_Topic_Source_Data_V2.0/data/ltf/"
    childuid2rsd = dict()
    for curr_path, directs, files in os.walk(LTF_PATH):
        if curr_path.endswith('.rsd'):
            for file in tqdm(files):
                child_uid = file[:9]
                file_path = os.path.join(curr_path, child_uid)
                rsd = open(file_path + '.rsd.txt', 'r')
                childuid2rsd[child_uid] = rsd.read()
        
    with open(f'{output_dir}/childuid2rsd.p', 'wb') as f:
        pickle.dump(childuid2rsd, f)
        
    # Create a mapping from root_uid to a list of all segments and their tokens.
    for curr_path, directs, files in tqdm(os.walk(LTF_PATH)):
        if curr_path.endswith('.ltf'):
            for file in files:
                ltfxml_path = os.path.join(curr_path, file)
                with open(ltfxml_path, 'r') as f:
                    data = f.read()
                    json_data = xmltodict.parse(data)
                    out_file_name = file_path[-17:].replace('.xml', '.json')
                    with open(f'{output_dir}/ltf.json/{out_file_name}', 'w') as f:
                        json.dump(json_data, f)
        
        
    # Create a mapping from (root_uid, seg_id) to the segment (i.e. the sentence).
    childuidsegid2segment = defaultdict()
    for curr_path, directs, files in os.walk(f'{output_dir}/ltf.json'):
        if curr_path.endswith('.json'):
            for file in tqdm(files):
                child_uid = file[:9]
                with open(f'{output_dir}/ltf.json/' + file, 'r') as json_file:
                    json_data = json.load(json_file)
                    assert child_uid == json_data['LCTL_TEXT']['DOC']['@id']
                    text = json_data['LCTL_TEXT']['DOC']['TEXT']
                    segments = text['SEG']
                    
                    # Case 1: The article contains multiple sentences (i.e. segments) stored in a list.
                    if type(segments) == list:
                        for seg in segments:
                            seg_id = seg['@id']
                            childuidsegid2segment[(child_uid, seg_id)] = seg['ORIGINAL_TEXT']
                            
                    # Case 2: The article contains only one sentence (type: dict) and 
                    #         is directly stored as the value of text['SEG'].
                    elif type(segments) == dict:
                        seg_id = segments['@id']
                        childuidsegid2segment[(child_uid, seg_id)] = segments['ORIGINAL_TEXT']
                        
                    else:
                        raise ValueError("Invalid text['SEG'] type!")
    childuidsegid2segment = dict(childuidsegid2segment)
        
    # Write the results to childuidsegid2segment.p.
    with open(f'{output_dir}/childuidsegid2segment.p', 'wb') as f:
        pickle.dump(childuidsegid2segment, f)
        
        
    # Create a mapping from child_uid to {segid: segment}.
    childuid2segments = defaultdict()
    for curr_path, directs, files in os.walk(f'{output_dir}/ltf.json'):
        if curr_path.endswith('.json'):
            for file in tqdm(files):
                child_uid = file[:9]
                with open(f'{output_dir}/ltf.json/' + file, 'r') as json_file:
                    json_data = json.load(json_file)
                    assert child_uid == json_data['LCTL_TEXT']['DOC']['@id']
                    text = json_data['LCTL_TEXT']['DOC']['TEXT']
                    segments = text['SEG']
                    segid2segment = defaultdict()
                    
                    # Case 1: The article contains multiple sentences (i.e. segments) stored in a list.
                    if type(segments) == list:
                        for seg in segments:
                            seg_id = seg['@id']
                            segid2segment[seg_id] = seg['ORIGINAL_TEXT']
                            
                    # Case 2: The article contains only one sentence (type: dict) and 
                    #         is directly stored as the value of text['SEG'].
                    elif type(segments) == dict:
                        seg_id = segments['@id']
                        segid2segment[seg_id] = segments['ORIGINAL_TEXT']
                        
                    else:
                        raise ValueError("Invalid text['SEG'] type!")
                    segid2segment = dict(segid2segment)
                    childuid2segments[child_uid] = segid2segment
    childuid2segments = dict(childuid2segments)
        
    # Write the results to childuid2segments.p.
    with open(f'{output_dir}/childuid2segments.p', 'wb') as f:
        pickle.dump(childuid2segments, f)
        
        
    # Create a mapping from child_uid to the language of the ltf document (<DOC lang>) and
    # a mapping from child_uid to raw_text_char_length.
    childuid2language = defaultdict()
    childuid2raw_text_char_length = defaultdict()
    for curr_path, directs, files in os.walk(f'{output_dir}/ltf.json/'):
        for file in tqdm(files):
            ltfjson_path = os.path.join(curr_path, file)
            with open(ltfjson_path, 'r') as file:
                data = json.load(file)
                child_uid = data['LCTL_TEXT']['DOC']['@id']
                childuid2language[child_uid] = data['LCTL_TEXT']['DOC']['@lang']
                childuid2raw_text_char_length[child_uid] = int(data['LCTL_TEXT']['DOC']['@raw_text_char_length'])
    childuid2language = dict(childuid2language)
    
    language_counter = Counter(childuid2language.values())
    print("language_counter:\n", language_counter, '\n')
    
    
    # Compute statistics.
    # Average length of raw text characters in each LTF document
    mean_raw_text_char_length = np.array(list(childuid2raw_text_char_length.values())).mean()
    print(mean_raw_text_char_length)

    # Variance of raw text character length
    var_raw_text_char_length = np.array(list(childuid2raw_text_char_length.values())).var()
    print(var_raw_text_char_length)

    # Standard deviation of raw text character length
    std_raw_text_char_length = np.array(list(childuid2raw_text_char_length.values())).std()
    print(std_raw_text_char_length)

    
    # Write the results to childuid2lang.p, childuid2raw_text_char_length.p and mean_raw_text_char_length.txt.
    with open(f'{output_dir}/childuid2doclang.p', 'wb') as file:
        pickle.dump(childuid2language, file)

    with open(f'{output_dir}/childuid2raw_text_char_length.p', 'wb') as file:
        pickle.dump(childuid2raw_text_char_length, file)

    with open(f'{output_dir}/statistics_raw_text_char_length.txt', 'w') as file:
        content = "Average length of raw text characters in each LTF document:\n" + str(mean_raw_text_char_length) + "\nVariance of raw text character length:\n" + str(var_raw_text_char_length) + "\nStandard deviation of raw text character length:\n" + str(std_raw_text_char_length)
        file.write(content)
    
    print("Done!")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        