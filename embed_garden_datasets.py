"""
@File  : embed_garden_datasets.py
@Date  : 11/23/26  12:37 PM
@License: See license file is in the root of the repository.
@Desc  :

Copyright (c) Aki Härmä, DACS, Maastricht University, 2026.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from test_model_garden import GardenTokenizer

GT = GardenTokenizer()

dpath = "/media/kakskyt/data/gardenFiles/GardenFiles23/GardenFiles23/"
outpath = "./embedded_garden"

os.makedirs(outpath, exist_ok=True)

files = [str(x) for x in Path(dpath).rglob("*_speechless.xlsx")]

for f in files:
    print(f"Processing {f}")
    dat = pd.read_excel(f)

    all_items = {}
    for c1 in range(dat.shape[0]):
        item = dat.loc[c1].to_dict()
        item["full_filename"] = f"{dpath}/{item["filename"]}"
        try:
            ids = GT.process_audio_file(item)
            item["pattern"] = ids
        except:
            item["pattern"] = ""
        all_items[c1] = item

    mfile = f.split("/")[-1].replace(".xlsx", ".pt")
    torch.save(all_items, f"{outpath}/{mfile}")

if __name__ == '__main__':
    pass
