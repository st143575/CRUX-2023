# CRUX-2023

skyline2023 at TAC 2023 CRUX.

## Setup

Prerequisite:
- python 3.11
- cuda 12.3

Install packages:
```bash
pip install -r requirements.txt
```
Download all the datasets to ```./datasets``` such that the directory has the following structure:  
*datasets*  
$\quad$|--*LDC*  
$\quad$| $\quad$|--*LDC2021E11_AIDA_Phase_3_Practice_Topic_Source_Data_V2.0*  
$\quad$| $\quad$|--*LDC2021E16_AIDA_Phase_3_TA3_Practice_Topic_Annotation_V5.1*  
$\quad$| $\quad$|--*LDC2023E10_SMKBP_2023_Claim_Frame_Evaluation_Source_Data*  
$\quad$|--*DWD_Overlay*  
$\quad$|--*CRUX2023_Task1_Evaluation_KBs*

## Data Preprocessing
Preprocess the raw datasets through the following steps:

1. Build mappings from the raw datasets by running:
  ```bash
  cd ./src/data_preprocessing/build_mappings/
  python build_mappings.py -i ../dataset -o ./output
  ```

2. Translate non-English segments to English by running:
  ```bash
  cd ./src/data_preprocessing/translate/
  python translate.py -i ../BuildMappings/outputs -o ./output -m nllb
  ```

3. Create translated documents (rsd files) by running:
  ```bash
  cd ./src/data_preprocessing/
  python create_translated_rsd.py -i ./translate/output -o ./output
  ```

4. Create CoT instruction data by running:
  ```bash
  cd ./src/data_preprocessing/
  python create_instruction_data.py -dp ../../datasets -trp ./translate/output -o ./data
  ```
