from abc import ABC, ABCMeta
from chromadb.api.models import Collection

class QueryVectorDatabase(ABC):
    def __init__(self,naics_collection:Collection):
        self.naics_collection = naics_collection
    
    def query(self,query_embeddings:list[float],num_returns:int):
        pass