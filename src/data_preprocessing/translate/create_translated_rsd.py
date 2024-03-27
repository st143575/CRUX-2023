"""
Create translated RSD by merging the segments in childuid2translatedsegments.p which belong to the same child_uid.
"""

import json, argparse
import dill as pickle
from pathlib import Path


def parse_arguments():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(description='Create translated rsd files.')
    parser.add_argument('-i', '--input_file_path', type=str, help="Path to the translated segments associated with each child_uid.")
    parser.add_argument('-o', '--output_dir', type=str, default='./output', help="Path to save the translated documents (rsd files).")
    return parser.parse_args()


def main():
    args = parse_arguments()
    input_fp = Path(args.input_file_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Input file path:", input_fp)
    print("Output directory:", output_dir)
    
    # Load childuid2translatedsegments.p
    childuid2translatedsegments = pickle.load(open(input_fp, 'rb'))
    
    # Merge segments of each child_uid to a single article (rsd).
    childuid2translatedrsd = dict()
    for child_uid, segments in childuid2translatedsegments.items():
        childuid2translatedrsd[child_uid] = '\n'.join(segments.values())

    # Get file name for the translated rsd.
    if 'trainval' in args.input_file_path:
        output_fn = "childuid2translatedrsd_trainval"
    elif 'eval' in args.input_file_path:
        output_fn = "childuid2translatedrsd_eval"
    else:
        raise ValueError("Input file name should contain either 'trainval' or 'eval'.")
        
    with open(f'{output_dir}/{output_fn}.json', 'w') as file:
        json.dump(childuid2translatedrsd, file)
        
    with open(f'{output_dir}/{output_fn}.p', 'wb') as file:
        pickle.dump(childuid2translatedrsd, file)
        
        
if __name__ == "__main__":
    main()
    