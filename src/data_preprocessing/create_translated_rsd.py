"""
Create translated RSD by merging the segments in childuid2translatedsegments.p which belong to the same child_uid.
"""

import os, json, argparse
import dill as pickle
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path


def parse_arguments():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(description='Create translated rsd files.')
    parser.add_argument('-i', '--input_dir', type=str, default='./translate/output', help="Path to the mappings created")
    parser.add_argument('-o', '--output_dir', type=str, default='./output', help="Path to save the translated texts")
    return parser.parse_args()


def main():
    args = parse_arguments()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Input directory:", input_dir)
    print("Output directory:", output_dir)
    
    # Load childuid2translatedsegments.p
    childuid2translatedsegments = pickle.load(open(f'{input_dir}/childuid2translatedsegments.p', 'rb'))
    
    # Merge segments of each child_uid to a single article (rsd).
    childuid2translatedrsd = dict()
    for child_uid, segments in childuid2translatedsegments.items():
        childuid2translatedrsd[child_uid] = '\n'.join(segments.values())
        
    with open(f'{output_dir}/childuid2translatedrsd.json', 'w') as file:
        json.dump(childuid2translatedrsd, file)
        
    with open(f'{output_dir}/childuid2translatedrsd.p', 'wb') as file:
        pickle.dump(childuid2translatedrsd, file)
        
        
if __name__ == "__main__":
    main()
    