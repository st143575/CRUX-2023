# CRUX-2023

skyline2023 at TAC 2023 CRUX.

## Setup


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
  python create_instruction_data.py -dp ../../datasets -trp ./translate/output -o ./output
  ```
