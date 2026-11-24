"""
@File  : test_model.py
@Date  : 10/16/26  8:25 AM
@License: See license file is in the root of the repository.
@Desc  :

Copyright (c) Aki Härmä, DACS, Maastricht University, 2026.
"""
import pandas as pd
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch
import soundfile as sf
import os
from pathlib import Path
from sklearn.cluster import KMeans, MiniBatchKMeans

def does_wav_exist(path):
    return (os.path.isfile(path))&(os.path.getsize(path)>1)

class GardenEmbedding:
    def __init__(self):
        model_id = "wav2vec2_pretraining/saved_model/epoch_4"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self.model = Wav2Vec2Model.from_pretrained(model_id)
        self.nomfs = 16000 # Nominal - not real FS

    def embed_audio_file(self, afile):
        wav = sf.read(afile)[0][:, 0] * 500.0
        inputs = self.feature_extractor(wav, sampling_rate=self.nomfs, return_tensors="pt")
        # Embedding
        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state

        return outputs.detach().numpy()


class GardenTokenizer:
    def __init__(self):
        self.embed_model = GardenEmbedding()
        self.cluster,_ = torch.load("clustering_garden.pt",weights_only=False)

    def process_audio_file(self, item):
        emb = self.embed_model.embed_audio_file(item["full_filename"])
        ids = self.cluster.predict(emb[0,:,:])
        return ids

"""
    Train the clustering using the pre-trained embedding model
"""
if __name__ == '__main__':
    train_data = pd.read_csv("./datasets/train.csv", sep="\t")

    cluster_model = MiniBatchKMeans(n_clusters=2048, random_state=0, n_init="auto")
    embed_model = GardenEmbedding()

    B = 1000
    N = int(train_data.shape[0]/B)

    for c1 in range(N):
        print(f"Processing {c1}/{N}")
        X = np.concatenate([embed_model.embed_audio_file(train_data.loc[c2, "filename"])
                            for c2 in range(c1*B, (c1+1)*B)])
        x = X.reshape([X.shape[0]*X.shape[1], X.shape[2]])
        pp = cluster_model.partial_fit(x)

    c = cluster_model.predict(x)
    torch.save([pp, c], "clustering_garden.pt")


