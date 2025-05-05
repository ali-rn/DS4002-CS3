# Importing & Processing Data from KiTS23

1. enter the following in your terminal. All data will be downloaded under a ```dataset/``` folder. See https://kits-challenge.org/kits23/ for more information on this dataset
```
git clone https://github.com/neheller/kits23
cd kits23
pip3 install -e .
kits23_download_data
cd ..
```
2. this dataset is the raw data. Use and tweak ```data_reorg.py```, ```data_split.py```, then ```data_slice.py``` to preprocess the data for model training. You can change ```data_reorg.py``` to increase or decrease the amount of cases you want to use.

## Importing Checkpoints for Testing
* if you have other metrics you want to do testing with, the checkpoints of weights for the other models have been provided.

## Importing Metrics Results
* the .csv's are provided for the results if you prefer to go straight to comparison
