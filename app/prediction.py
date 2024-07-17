from chromadb.api.models import Collection
from app.abstractclasses import PredictionAbstract
from app.gliner import get_entities, load_model
from app.vectordb import get_embedding_fn
from functools import partial
from typing import Tuple, List
from dataclasses import dataclass

@dataclass
class QueryResult:
    industry: str
    naics_code: str
    type: str

class Prediction(PredictionAbstract):
    def __init__(
        self, naics_collection: Collection, threshold: float = 0.2,
    ):
        """Constructor method for Prediction object to get the NAICS codes

        Args:
            naics_collection (Collection): NAICS Vector Database collection
            threshold (float, optional): threshold for GliNER. Defaults to 0.2.
        """
        self.naics_collection = naics_collection
        self.threshold = threshold
        self.gliner_model = load_model()
        self.gliner_partial = partial(get_entities, gliner_model=self.gliner_model)
        self.emb_fn = get_embedding_fn()

    def _get_industry_names_code(
        self, relevant_docs: dict
    )->List[QueryResult]:
        """Return the NAICS code and industry names

        Args:
            relevant_docs (dict): Relevant docs form vector database 

        Returns:
            List[QueryResult]: List of NAICS code and industry names as QueryResult
        """
        industry_names_code = []

        for doc in relevant_docs['metadatas'][0]:
            if 'PARENT NAME' in doc:
                industry_name = "PARENT NAME"
                type_ = "PARENT"
            else:
                industry_name = "CHILD NAME"
                type_ = "CHILD"
            industry_names_code.append(QueryResult(industry=doc[industry_name],naics_code=doc['NAICS CODE'],type=type_))
        return industry_names_code

    def pipeline(self, question: str,n_results:int=1)->Tuple[List[Tuple],List[QueryResult]]:
        """Pipeline object to get the results from a question

        Args:
            question (str): Question to get NAICS codes
            n_results (int, optional): Number of vetor database results. Defaults to 1.

        Returns:
            Tuple[List[Tuple],List[QueryResult]]: Span and QueryResults are returned
        """
        entities = self.gliner_partial(question=question, threshold=self.threshold)

        assert (
            len(entities) >= 1
        ), "Expected atleast one entity. Try changing the threshold or the labels for better NER"

        ner_text_list = [entity["text"] for entity in entities]
        ner_entities_span = [(entity['start'],entity['end']) for entity in entities]
        ner_embeddings = self.emb_fn(ner_text_list)

        # parent_relevant_docs = self.query_database(ner_embeddings, "parent")
        # parent_industry_names,parent_industry_codes = self._get_industry_names_code(
        #     parent_relevant_docs, "parent"
        # )

        # child_relevant_docs = self.query_database(
        #     ner_embeddings, "child", parent_industry_names
        # )
        # child_industry_names, child_industry_codes  = self._get_industry_names_code(
        #     child_relevant_docs, "child"
        # )
        relevant_docs = self.naics_collection.query(
                query_embeddings=ner_embeddings,
                n_results=n_results,
            )
        
        industry_names_code = self._get_industry_names_code(relevant_docs)

        return ner_entities_span,industry_names_code
    
    # def query_database(
    #     self,
    #     query_embeddings: list[list[float]],
    #     parent_or_child: Literal["parent", "child"],
    #     parent_names: List[str] = "",
    # ):

    #     if parent_or_child == "parent":
    #         relevant_docs = self.naics_collection.query(
    #             query_embeddings=query_embeddings,
    #             where={"TYPE": "PARENT"},
    #             n_results=self.n_results,
    #             # include = ["metadatas", "documents", "embeddings"]
    #         )
    #         return relevant_docs

    #     elif parent_or_child == "child":
    #         assert (
    #             parent_names != []
    #         ), "Parent name is required for querying child documents"
    #         relevant_docs = self.naics_collection.query(
    #             query_embeddings=query_embeddings,
    #             where={
    #                 "$and": [
    #                     {"TYPE": {"$eq": "CHILD"}},
    #                     {"PARENT": {"$in": parent_names}},
    #                 ]
    #             },
    #             n_results=self.n_results,
    #         )
    #         return relevant_docs

    # def _get_industry_names_code(
    #     self, relevant_docs, parent_or_child: Literal["parent", "child"]
    # ):

    #     if parent_or_child == "parent":
    #         industry_name = "PARENT NAME"
    #     elif parent_or_child == "child":
    #         industry_name = "CHILD NAME"

    #     return [doc[industry_name] for doc in relevant_docs["metadatas"][0]], [
    #         doc["NAICS CODE"] for doc in relevant_docs["metadatas"][0]
    #     ]