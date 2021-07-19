# BERT-based model utils for Sentiment Attitude Extraction task

<p align="center">
    <img src="logo.png"/>
</p>
    
This repository is BERT-based model service for Sentiment Attitude Extraction,
based on [AREkit](https://github.com/nicolay-r/AREkit) framework.

## Utils List

* [Data Serialization](#usage-data-serialization)
* [Results Evaluation](#usage-results-evaluation)

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

For flags meanings please proceed with [this section](#script-arguments-manual)

### Usage: Results Evaluation

Proceed with the [following](results_evaluation.ipynb) notebook.

### Script Arguments Manual

Common flags:
* `--experiment` -- is an experiment which could be as follows:
    * `rsr` -- supervised learning + evaluation within [RuSentRel](https://github.com/nicolay-r/RuSentRel) collection;
    * `ra` -- pretraining with [RuAttitudes](https://github.com/nicolay-r/RuAttitudes) collection;
    * `rsr+ra` -- combined training within RuSentRel and RuAttitudes and evalut.
* `--cv_count` -- data folding mode:
    * `1` -- predefined docs separation onto TRAIN/TEST (RuSentRel);
    * `k` -- CV-based folding onto `k`-folds; (`k=3` supported);
* `--frames_versions` -- RuSentiFrames collection version:
    * `v2.0` -- RuSentiFrames-2.0;
* `--ra_ver` -- RuAttitudes version, if collection is applicable (`ra` or `rsr+ra` experiments):
    * `v1_2` -- RuAttitudes-1.0 [paper](https://www.aclweb.org/anthology/R19-1118/);
    * `v2_0_base`;
    * `v2_0_large`;
    * `v2_0_base_neut`;
    * `v2_0_large_neut`;
* `--bert-input-fmt` -- supported input formatters
    * `c_m` -- single input (TEXT_A);
    * `nli_m` -- TEXT_A + context in between of the attitude participants (TEXB_B);
    * `qa_m` -- TEXT_A + question.

### References

> TODO. To be updated.
