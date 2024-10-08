
## Datasets
* You can download multi-hop datasets (MuSiQue, HotpotQA, and 2WikiMultiHopQA) from https://github.com/StonyBrookNLP/ircot.
```bash
# Download the preprocessed datasets for the test set.
$ bash ./download/processed_data.sh
# Prepare the dev set, which will be used for training our query complexity classfier.
$ bash ./download/raw_data.sh
$ python processing_scripts/subsample_dataset_and_remap_paras.py musique dev_diff_size 500
$ python processing_scripts/subsample_dataset_and_remap_paras.py hotpotqa dev_diff_size 500
$ python processing_scripts/subsample_dataset_and_remap_paras.py 2wikimultihopqa dev_diff_size 500
