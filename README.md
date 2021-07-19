# BERT-based model utils for Sentiment Attitude Extraction task

<p align="center">
    <img src="logo.png"/>
</p>
    
This repository is BERT-based model service for Sentiment Attitude Extraction,
based on [AREkit](https://github.com/nicolay-r/AREkit) framework.

## Utils List

* [Data Serialization](#data-serialization)
* [Data Converter for Attention Analysis](#data-converter-for-attention-analysis)
* [Results Evaluation](#results-evaluation)

## Dependencies

* Python 2.7.9
* AREkit == 0.20.5
* tqdm

## Installation

AREkit repository:
```shell script
# Clone repository in local folder of the currect project. 
git clone -b 0.20.5-rc https://github.com/nicolay-r/AREkit ../arekit
# Install dependencies.
pip install -r arekit/requirements.txt
```

### Usage: Data Serialization

Using `run_serialization.sh` in order to prepare data for a particular experiment:

```shell script
python run_serialization.py 
    --cv-count 3 --frames-version v2_0 
    --experiment rsr+ra --labels-count 3 --ra-ver v1_0
    --entity-fmt rus-simple --balance-samples True
    --bert-input-fmt c_m
```

Supported collections:
* RuSentRel
* RuAttitudes

### Results Evaluation

Proceed with the [following](results_evaluation.ipynb) notebook.

### References

> TODO. To be updated.
