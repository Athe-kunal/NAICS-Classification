from chromadb.api.models import Collection
from app.abstractclasses import PredictionAbstract
from app.gliner import get_entities, load_model
from app.vectordb import get_embedding_fn
from functools import partial
from typing import Literal


class Prediction(PredictionAbstract):
    def __init__(self, naics_collection: Collection, threshold: float = 0.2):
        self.naics_collection = naics_collection
        self.threshold = threshold
        self.gliner_model = load_model()
        self.gliner_partial = partial(get_entities, gliner_model=self.gliner_model)
        self.emb_fn = get_embedding_fn()

    def query_database(
        self,
        query_embeddings: list[list[float]],
        parent_or_child: Literal["parent", "child"],
        parent_names: list = [],
    ):

        if parent_or_child == "parent":
            relevant_docs = self.naics_collection.query(
                query_embeddings=query_embeddings, where={"TYPE": "PARENT"}, n_results=1,
                include = ["metadatas", "documents", "embeddings"]
            )
            return relevant_docs

        elif parent_or_child == "child":
            relevant_docs = self.naics_collection.query(
                query_embeddings=query_embeddings,
                where={
                    "$and": [
                        {"TYPE": {"$eq": "CHILD"}},
                        {"metadata_field": {"$in": parent_names}},
                    ]
                },
                n_results=1,
            )
            return relevant_docs

    def pipeline(self, question: str):

        entities = self.gliner_partial(question=question, threshold=self.threshold)

        assert (
            len(entities) >= 1
        ), "Expected atleast one entity. Try changing the threshold or the labels for better NER"

        query_texts = [entity["text"] for entity in entities]
        query_embeddings = self.emb_fn(query_texts)

        return query_embeddings
