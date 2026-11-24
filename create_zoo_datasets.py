"""
@File  : create_zoo_datasets.py
@Date  : 10/9/26  3:32 PM
@License: See license file is in the root of the repository.
@Desc  :

Copyright (c) Aki Härmä, DACS, Maastricht University, 2026.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def does_wav_exist(path):
    return os.path.isfile(path)

dpath = "/media/kakskyt/data/zoodata/er_path"
mpath = "../spatial-stereo-model/new_meta"
outpath = "./datasets"
os.makedirs(outpath, exist_ok=True)

table = pd.read_excel("all_detections_species_aviaries.xlsx")
sel_species = ['Bostrychia hagedash', 'Geronticus eremita', 'Recurvirostra avosetta',
               'Eudocimus ruber ', 'Eudocimus ruber', 'Sagittarius serpentarius',
               'Lamprotornis superbus', 'Musophaga violacea']
table = table[table["species"].apply(lambda x: x in sel_species)].reset_index(drop=True)

out_table = []
for c1 in range(table.shape[0]):
    aviary_name = table.loc[c1, "aviary"][:-10]
    meta = pd.read_excel(f"{mpath}/{table.loc[c1, 'aviary']}")
    meta = meta[meta["reasoned_species"].apply(lambda x: x in sel_species)].reset_index(drop=True)
    meta["duration"] = 3.25
    meta["filename"] = dpath+"/"+meta['filename']
    out_table.append(meta[["filename","duration"]].copy())

full_list = pd.concat(out_table, ignore_index=True)
train, test = train_test_split(full_list, test_size=0.1)

train.to_csv(f"{outpath}/train.csv", index=False, sep="\t")
test.to_csv(f"{outpath}/test.csv", index=False, sep="\t")



if __name__ == '__main__':
    pass
