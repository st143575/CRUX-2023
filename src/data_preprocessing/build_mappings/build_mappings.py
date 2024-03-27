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


def parentuid2childuid(parent_children_df):
    """
    Build a mapping from parent_uid (i.e. root_uid) to child_uid, regardless of the type of the file.
    """
    parentuid2childuid2assettype = defaultdict()
    for _, row in parent_children_df.iterrows():
        parentuid2childuid2assettype[row['parent_uid']] = str(row['child_uid'])
    parentuid2childuid2assettype = dict(parentuid2childuid2assettype)
    return parentuid2childuid2assettype


def parentuid2ltfchilduid(parent_children_df):
    """
    Build a mapping from parent_uid to child_uid of the ltf file belonging to the HTML document represented by that parent_uid.
    """
    parentuid2ltfchilduid = defaultdict()
    for _, row in parent_children_df.iterrows():
        if row['child_asset_type'] == '.ltf.xml':
            parentuid2ltfchilduid[row['parent_uid']] = str(row['child_uid'])
    parentuid2ltfchilduid = dict(parentuid2ltfchilduid)
    return parentuid2ltfchilduid


def childuid_assettype2parentuid(parent_children_df):
    """
    Build a mapping from (child_uid, child_asset_type) to parent_uid.
    """
    childuid_assettype2parentuid = defaultdict(list)
    for _, row in parent_children_df.iterrows():
        childuid_assettype2parentuid[(str(row['child_uid']), row['child_asset_type'])].append(row['parent_uid'])
    childuid_assettype2parentuid = dict(childuid_assettype2parentuid)
    return childuid_assettype2parentuid


def childuid2rsd(ltf_path):
    """
    # Create a mapping from child_uid (equivalent to the file name) to the content of the rsd file, i.e. 
    # the raw text of the article.
    """
    childuid2rsd = dict()
    for curr_path, directs, files in os.walk(ltf_path):
        if curr_path.endswith('.rsd'):
            for file in tqdm(files):
                child_uid = file[:9]
                file_path = os.path.join(curr_path, child_uid)
                rsd = open(file_path + '.rsd.txt', 'r')
                childuid2rsd[child_uid] = rsd.read()
    childuid2rsd = dict(childuid2rsd)
    return childuid2rsd


def rootuid2segments_tokens(ltf_path):
    """
    Create a mapping from root_uid to a list of all segments and their tokens.
    """
    for curr_path, directs, files in tqdm(os.walk(ltf_path)):
        if curr_path.endswith('.ltf'):
            for file in files:
                ltfxml_path = os.path.join(curr_path, file)
                with open(ltfxml_path, 'r') as f:
                    data = f.read()
                    json_data = xmltodict.parse(data)
                    out_file_name = ltfxml_path[-17:].replace('.xml', '.json')
                    yield json_data, out_file_name

                    
def childuidsegid2segment(inp_dir):
    """
    Build a mapping from child_uid and segment_id to the segment (i.e. the sentence).
    """
    childuidsegid2segment = defaultdict()
    for curr_path, directs, files in os.walk(inp_dir):
        if curr_path.startswith('output/ltf_json_'):
            for file in tqdm(files):
                child_uid = file[:9]
                with open(f'{inp_dir}/' + file, 'r') as json_file:
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
    return childuidsegid2segment


def childuid2segments(inp_dir, split):
    """
    Create a mapping from child_uid to {segid: segment}.
    """
    childuid2segments = defaultdict()
    for curr_path, directs, files in os.walk(f'{inp_dir}/ltf_json_{split}'):
        if curr_path.startswith('output/ltf_json_'):
            for file in tqdm(files):
                child_uid = file[:9]
                with open(f'{inp_dir}/ltf_json_{split}/' + file, 'r') as json_file:
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
    return childuid2segments


def childuid2doclang(inp_dir, split):
    """
    Create a mapping from child_uid to the language of the ltf document (<DOC lang>) and a mapping from child_uid to raw_text_char_length.
    """
    childuid2language = defaultdict()
    childuid2raw_text_char_length = defaultdict()
    for curr_path, directs, files in os.walk(f'{inp_dir}/ltf_json_{split}/'):
        for file in tqdm(files):
            ltfjson_path = os.path.join(curr_path, file)
            with open(ltfjson_path, 'r') as file:
                data = json.load(file)
                child_uid = data['LCTL_TEXT']['DOC']['@id']
                childuid2language[child_uid] = data['LCTL_TEXT']['DOC']['@lang']
                childuid2raw_text_char_length[child_uid] = int(data['LCTL_TEXT']['DOC']['@raw_text_char_length'])
    childuid2language = dict(childuid2language)
    
    language_counter = Counter(childuid2language.values())
    print(f"language_counter_{split}:\n", language_counter, '\n')
    return childuid2language, childuid2raw_text_char_length


def compute_statistics(childuid2raw_text_char_length):
    """
    Compute:
        - Average length of raw text characters in each LTF document;
        - Variance of raw text character length;
        - Standard deviation of raw text character length.
    """
    # Average length of raw text characters in each LTF document
    mean_raw_text_char_length = np.array(list(childuid2raw_text_char_length.values())).mean()
    print(mean_raw_text_char_length)

    # Variance of raw text character length
    var_raw_text_char_length = np.array(list(childuid2raw_text_char_length.values())).var()
    print(var_raw_text_char_length)

    # Standard deviation of raw text character length
    std_raw_text_char_length = np.array(list(childuid2raw_text_char_length.values())).std()
    print(std_raw_text_char_length)

    return mean_raw_text_char_length, var_raw_text_char_length, std_raw_text_char_length


def main():
    args = parse_arguments()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Input directory:", input_dir)
    print("Output directory:", output_dir)
    
    LDC2021E11_PATH = f'{input_dir}/LDC/LDC2021E11_AIDA_Phase_3_Practice_Topic_Source_Data_V2.0'
    LDC2021E16_PATH = f'{input_dir}/LDC/LDC2021E16_AIDA_Phase_3_TA3_Practice_Topic_Annotation_V5.1'
    LDC2023E10_PATH = f'{input_dir}/LDC/LDC2023E10_SMKBP_2023_Claim_Frame_Evaluation_Source_Data'
    DWD_PATH = f'{input_dir}/DWD_Overlay/'
    
    # Load parent_children.tab (train_val)
    parent_children_df_trainval = pd.read_csv(f'{LDC2021E11_PATH}/docs/parent_children.tab', sep='\t')

    # Load parent_children.tab (eval)
    parent_children_df_eval = pd.read_csv(f'{LDC2023E10_PATH}/docs/parent_children.tab', sep='\t')


    # Build a mapping from parent_uid (i.e. root_uid) to child_uid, regardless of the type of the file.
    # From train_val data.
    parentuid2childuid2assettype_trainval = parentuid2childuid(parent_children_df_trainval)
    with open(f"{output_dir}/parentuid2childuid2assettype_trainval.p", 'wb') as file:
            pickle.dump(parentuid2childuid2assettype_trainval, file)

    # Build a mapping from parent_uid (i.e. root_uid) to child_uid, regardless of the type of the file.
    # From eval data.
    parentuid2childuid2assettype_eval = parentuid2childuid(parent_children_df_eval)
    with open(f"{output_dir}/parentuid2childuid2assettype_eval.p", 'wb') as file:
            pickle.dump(parentuid2childuid2assettype_eval, file)
              
    
    # Build a mapping from parent_uid to child_uid of the ltf file belonging to the HTML document represented by that parent_uid.
    # From train_val data.
    parentuid2ltfchilduid_trainval = parentuid2ltfchilduid(parent_children_df_trainval)
    with open(f'{output_dir}/parentuid2ltfchilduid_trainval.p', 'wb') as file:
        pickle.dump(parentuid2ltfchilduid_trainval, file)


    # Build a mapping from parent_uid to child_uid of the ltf file belonging to the HTML document represented by that parent_uid.
    # From eval data.
    parentuid2ltfchilduid_eval = parentuid2ltfchilduid(parent_children_df_eval)
    with open(f'{output_dir}/parentuid2ltfchilduid_eval.p', 'wb') as file:
        pickle.dump(parentuid2ltfchilduid_eval, file)
        
        
    # Build a mapping from (child_uid, child_asset_type) to parent_uid.
    # From train_val data.
    childuid_assettype2parentuid_trainval = childuid_assettype2parentuid(parent_children_df_trainval)
    with open(f'{output_dir}/childuid_assettype2parentuid_trainval.p', 'wb') as file:
        pickle.dump(childuid_assettype2parentuid_trainval, file)


    # Build a mapping from (child_uid, child_asset_type) to parent_uid.
    # From eval data.
    childuid_assettype2parentuid_eval = childuid_assettype2parentuid(parent_children_df_eval)
    with open(f'{output_dir}/childuid_assettype2parentuid_eval.p', 'wb') as file:
        pickle.dump(childuid_assettype2parentuid_eval, file)
    
    
    # Create a mapping from child_uid (equivalent to the file name) to the content of the rsd file, i.e. 
    # the raw text of the article.
    # From train_val data.
    LTF_PATH_trainval = f"{LDC2021E11_PATH}/data/ltf/"
    childuid2rsd_trainval = childuid2rsd(ltf_path=LTF_PATH_trainval)
    with open(f'{output_dir}/childuid2rsd_trainval.p', 'wb') as f:
        pickle.dump(childuid2rsd_trainval, f)


    # Create a mapping from child_uid (equivalent to the file name) to the content of the rsd file, i.e. 
    # the raw text of the article.
    # From eval data.
    LTF_PATH_eval = f"{LDC2023E10_PATH}/data/ltf/"
    childuid2rsd_eval = childuid2rsd(ltf_path=LTF_PATH_eval)
    with open(f'{output_dir}/childuid2rsd_eval.p', 'wb') as f:
        pickle.dump(childuid2rsd_eval, f)

        
    # Create a mapping from root_uid to a list of all segments and their tokens.
    # From train_val data.
    os.makedirs(f'{output_dir}/ltf_json_trainval', exist_ok=True)
    for json_data_trainval, out_file_name_trainval in rootuid2segments_tokens(ltf_path=LTF_PATH_trainval):
        with open(f'{output_dir}/ltf_json_trainval/{out_file_name_trainval}', 'w') as f:
            json.dump(json_data_trainval, f)


    # Create a mapping from root_uid to a list of all segments and their tokens.
    # From eval data.
    os.makedirs(f'{output_dir}/ltf_json_eval', exist_ok=True)
    for json_data_eval, out_file_name_eval in rootuid2segments_tokens(ltf_path=LTF_PATH_eval):
        with open(f'{output_dir}/ltf_json_eval/{out_file_name_eval}', 'w') as f:
            json.dump(json_data_eval, f)
        
        
    # Create a mapping from (root_uid, seg_id) to the segment (i.e. the sentence).
    # From train_val data.
    childuidsegid2segment_trainval = childuidsegid2segment(f'{output_dir}/ltf_json_trainval')
    with open(f'{output_dir}/childuidsegid2segment_trainval.p', 'wb') as f:
        pickle.dump(childuidsegid2segment_trainval, f)


    # Create a mapping from (root_uid, seg_id) to the segment (i.e. the sentence).
    # From eval data.
    childuidsegid2segment_eval = childuidsegid2segment(f'{output_dir}/ltf_json_eval')
    with open(f'{output_dir}/childuidsegid2segment_eval.p', 'wb') as f:
        pickle.dump(childuidsegid2segment_eval, f)
        
        
    # Create a mapping from child_uid to {segid: segment}.
    # From train_val data.
    childuid2segments_trainval = childuid2segments(inp_dir=output_dir, split='trainval')
    with open(f'{output_dir}/childuid2segments_trainval.p', 'wb') as f:
        pickle.dump(childuid2segments_trainval, f)


    # Create a mapping from child_uid to {segid: segment}.
    # From eval data.
    childuid2segments_eval = childuid2segments(inp_dir=output_dir, split='eval')
    with open(f'{output_dir}/childuid2segments_eval.p', 'wb') as f:
        pickle.dump(childuid2segments_eval, f)
    
        
    # Create a mapping from child_uid to the language of the ltf document (<DOC lang>) and
    # a mapping from child_uid to raw_text_char_length.
    # From train_val data.
    childuid2language_trainval, childuid2raw_text_char_length_trainval = childuid2doclang(inp_dir=output_dir, split='trainval')
    with open(f'{output_dir}/childuid2doclang_trainval.p', 'wb') as f:
        pickle.dump(childuid2language_trainval, f)

    with open(f'{output_dir}/childuid2raw_text_char_length_trainval.p', 'wb') as file:
        pickle.dump(childuid2raw_text_char_length_trainval, file)


    # Create a mapping from child_uid to the language of the ltf document (<DOC lang>) and
    # a mapping from child_uid to raw_text_char_length.
    # From eval data.
    childuid2language_eval, childuid2raw_text_char_length_eval = childuid2doclang(inp_dir=output_dir, split='eval')
    with open(f'{output_dir}/childuid2doclang_eval.p', 'wb') as f:
        pickle.dump(childuid2language_eval, f)

    with open(f'{output_dir}/childuid2raw_text_char_length_eval.p', 'wb') as file:
        pickle.dump(childuid2raw_text_char_length_eval, file)
    
    
    # Compute statistics of train_val data.
    mean_raw_text_char_length_trainval, var_raw_text_char_length_trainval, std_raw_text_char_length_trainval = compute_statistics(childuid2raw_text_char_length_trainval)

    with open(f'{output_dir}/statistics_raw_text_char_length_trainval.txt', 'w') as file:
        content = "Average length of raw text characters in each LTF document:\n" + str(mean_raw_text_char_length_trainval) + \
        "\nVariance of raw text character length:\n" + str(var_raw_text_char_length_trainval) + \
        "\nStandard deviation of raw text character length:\n" + str(std_raw_text_char_length_trainval)
        file.write(content)


    # Compute statistics of eval data.
    mean_raw_text_char_length_eval, var_raw_text_char_length_eval, std_raw_text_char_length_eval = compute_statistics(childuid2raw_text_char_length_eval)

    with open(f'{output_dir}/statistics_raw_text_char_length_eval.txt', 'w') as file:
        content = "Average length of raw text characters in each LTF document:\n" + str(mean_raw_text_char_length_eval) + \
        "\nVariance of raw text character length:\n" + str(var_raw_text_char_length_eval) + \
        "\nStandard deviation of raw text character length:\n" + str(std_raw_text_char_length_eval)
        file.write(content)
    
    print("Done!")
    
    
if __name__ == "__main__":
    main()
    