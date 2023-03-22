# Silver Data for Coreference Resolution in Ukrainian: Translation, Alignment, and Projection

This repository contains the following:

- the code to build the Ukrainian silver data for coreference resolution based on OntoNotes 5.0
- the manual translation of Winograd Schema Challenge dataset into Ukrainian

## Ukrainian OntoNotes Dataset

The experiments were conducted using OntoNotes 5.0 data. The corpus can be downloaded [here](https://catalog.ldc.upenn.edu/LDC2013T19); registration needed.

### Preprocessing

 The provided code expects data in `jsonlines` format, so some preprocessing is necessary.

1. Extract OntoNotes 5.0 arhive. In case it's in the repo's root directory:
  
        tar -xzvf ontonotes-release-5.0_LDC2013T19.tgz
        
2. Switch to Python 2.7 environment (where `python` would run 2.7 version). This is necessary for CoNLL scripts to run correctly. To do it with conda:

        conda create -y --name py27 python=2.7 && conda activate py27
        
3. Run the CoNLL data preparation scripts:

        sh preprocessing/get_conll_data.sh ontonotes-release-5.0 ontonotes-ua
        
4. Download the CoNLL scorers and Stanford Parser:

        sh preprocessing/get_third_party.sh
        
5. Prepare your environment. To do it with conda:

        conda create -y --name ua-coref-data python=3.7 openjdk perl
        conda activate ua-coref-data
        python -m pip install -r requirements.txt
        
6. Build the corpus in `jsonlines` format:

        python preprocessing/convert_to_jsonlines.py ontonotes-ua/conll-2012/ --out-dir ontonotes-ua
        
### Building the silver Ukrainian version

Run the scripts to translate the sentences, align the mentions, and project the annotations from English to Ukrainian:

        python scripts/build_silver_data.py -train -dev -test

Processing the whole corpus may take a while because of the current logic behind MT model usage, so you may exclude some splits if necessary.

The machine translation model can be specified using the `--translation_model` flag. Note that in our experiments, `Helsinki-NLP/opus-mt-en-uk` model was used, and alignment is based on the cross-attention of the 0-th head of the 1-st layer. Using a different model may require changing this as well.



### Statistics

The dataset contains:

| Split     | Documents | Sentences | Tokens    | Mentions | Clusters |
| --------- | --------- | --------- | --------- | -------- | -------- |
| train     | 2,802     | 75,187    | 1,158,965 | 161,010  | 35,025   |
| dev       | 343       | 9,603     | 146,210   | 20,168   | 4,533    |
| test      | 348       | 9,479     | 151,542   | 20,522   | 4,513    |
| **TOTAL** | 3,493     | 94,269    | 1,456,717 | 201,700  | 44,071   |

## Ukrainian WSC Dataset

`wsc-ua` contains manual translations of 263 Winograd schemas from the WSC dataset in `csv` and `jsonlines` formats.

### Format

- `text` - the Winograd schema in Ukrainian, tokenized
- `options` - the two entity options that the pronoun may be referring to
- `label` - the index of the correct option in `options`
- `pronoun` - the pronoun in the sequence to be resolved
- `pronoun_loc` - the index of the ambiguous pronoun in `text`

No equivalent translations were found for 22 original schemas, so they were excluded:

87-88, 217-218, 221-222, 231-232, 233-234, 237-238, 243-244, 245-246, 247-248, 274-275, 276-277


## Contributing

Data and code improvements are welcome. Please submit a pull request.

## Citation

TBA

## Contacts

pavlo.kuchmiichuk@rochester.edu
