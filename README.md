# CRUX-2023

skyline2023 at TAC 2023 CRUX.

## Setup


## Data Preprocessing
- First, build mappings from the raw datasets by running:
  ```bash
  cd ./src/data_preprocessing/build_mappings/
  python build_mappings.py -i ../dataset -o ./output
  ```

- Then, translate non-English segments to English by running:
  ```bash
  cd ./src/data_preprocessing/translate/
  python translate.py -i ../BuildMappings/outputs -o ./output -m nllb
  ```
