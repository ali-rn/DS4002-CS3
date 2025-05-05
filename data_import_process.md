# Importing & Processing Data from KiTS23

**Tip: we imported the data into VSCode**

1. enter the following in your terminal. All data will be downloaded under a ```dataset/``` folder. See https://kits-challenge.org/kits23/ for more information on this dataset
```
git clone https://github.com/neheller/kits23
cd kits23
pip3 install -e .
kits23_download_data
cd ..
```
2. this dataset is the raw data, which will be split 64/16/20 (train/validate/test) and cleaned automatically by each model script
