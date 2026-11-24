"""
@File  : create_zoo_datasets.py
@Date  : 10/9/26  3:32 PM
@License: See license file is in the root of the repository.
@Desc  :

Copyright (c) Aki Härmä, DACS, Maastricht University, 2026.
"""
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def does_wav_exist(path):
    return (os.path.isfile(path))&(os.path.getsize(path)>1)

dpath = "/media/kakskyt/data/gardenFiles/GardenFiles23/GardenFiles23/"
outpath = "./garden_datasets"
os.makedirs(outpath, exist_ok=True)

files = list(Path(dpath).rglob("*_speechless.xlsx"))

bad_folder = "garden_20250510"
files = [f for f in files if str(f).find(bad_folder)==-1]



out_table = []
for f in files:
    try:
        d = pd.read_excel(f)
        d = d[d["filename"].apply(lambda x: does_wav_exist(dpath+"/"+x))]
        meta = d[["filename"]].copy()
        meta["duration"] = 132000/16000
        meta["filename"] = dpath+"/"+meta['filename']
        out_table.append(meta[["filename","duration"]].copy())
    except:
        pass

full_list = pd.concat(out_table, ignore_index=True)
full_list = full_list[full_list["filename"].apply(lambda x: does_wav_exist(x))].reset_index(drop=True)
train, test = train_test_split(full_list, test_size=0.1)

train.to_csv(f"{outpath}/train.csv", index=False, sep="\t")
test.to_csv(f"{outpath}/test.csv", index=False, sep="\t")



if __name__ == '__main__':
    pass
