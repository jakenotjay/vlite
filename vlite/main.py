import numpy as np
from uuid import uuid4
from .model import EmbeddingModel
from .utils import chop_and_chunk, cos_sim
from typing import List

class VLite:
    '''
    vlite is a simple vector database that stores vectors in a numpy array.
    '''

    texts: List[str] = []
    """List of text chunks that have been memorized."""

    metadata: dict = {}
    """Metadata for each text chunk."""

    vectors = None
    """Numpy array of vectors."""

    model = None
    """Embedding model."""

    lookup_table: dict = {}
    """Lookup table of text chunk ids to metadata indices."""

    def __init__(self, collection='vlite.npz',device='mps',model_name=None):
        self.collection = collection
        self.device = device
        self.model = EmbeddingModel() if model_name is None else EmbeddingModel(model_name)
        try:
            with np.load(self.collection, allow_pickle=True) as data:
                self.texts = data['texts'].tolist()
                self.metadata = data['metadata'].tolist()
                self.vectors = data['vectors']
                self.lookup_table = data['lookup_table'].tolist()
        except FileNotFoundError:
            self.texts = []
            self.metadata = {}
            self.vectors = np.empty((0, self.model.dimension))
            self.lookup_table = {}
    
    def add_vector(self, vector):
        self.vectors = np.vstack((self.vectors, vector))

    def get_similar_vectors(self, vector, top_k=5):
        sims = cos_sim(vector, self.vectors)
        sims = sims[0]
        # print("[get_similar_vectors] Sims:", sims.shape)
        top_k_idx = np.argsort(sims)[::-1][:top_k]
        # print("[get_similar_vectors] Top k idx:", top_k_idx)
        # print("[get_similar_vectors] Top k sims:", sims[top_k_idx])
        return top_k_idx, sims[top_k_idx]

    def memorize(self, text, id=None, metadata=None):
        id = id or str(uuid4())
        chunks = chop_and_chunk(text)
        encoded_data = self.model.embed(texts=chunks, device=self.device)
        self.vectors = np.vstack((self.vectors, encoded_data))

        # add the metadata to the metadata dict
        self.metadata[id] = metadata or {}

        for chunk in chunks:
            self.texts.append(chunk)
            # index of the last text chunk
            idx = len(self.texts) - 1
            self.lookup_table[idx] = id

        self.save()
        return id, self.vectors
    
    def chunk_id_to_metadata(self, chunk_id):
        return self.metadata[self.lookup_table[chunk_id]]

    def remember(self, text=None, id=None, top_k=5):
        if id:
            return self.chunk_id_to_metadata(id)
        if text:

            sims = cos_sim(self.model.embed(texts=text, device=self.device) , self.vectors)
            print("[remember] Sims:", sims.shape)
            sims = sims[0]

            # Use np.argpartition to partially sort only the top 5 values
            top_5_idx = np.argpartition(sims, -top_k)[-top_k:]  

            # Use np.argsort to sort just those top 5 indices
            top_5_idx = top_5_idx[np.argsort(sims[top_5_idx])[::-1]]

            # print("[remember] Top k sims:", sims[top_5_idx])
            return [self.texts[idx] for idx in top_5_idx], sims[top_5_idx], top_5_idx
            
    def save(self):
        with open(self.collection, 'wb') as f:
            np.savez(f, texts=self.texts, metadata=self.metadata, vectors=self.vectors, lookup_table=self.lookup_table)
