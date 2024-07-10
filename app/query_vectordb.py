
from chromadb.api.models import Collection
from app.abstractclasses import QueryVectorDatabase


class ParentQueryVectorDB(QueryVectorDatabase):
    """Query Vector Database for parent results

    Args:
        naics_collection (Collection): ChromaDB Vector database collection
        num_returns (int): Number of relevant parent documents to return
    """
    def __init__(self,naics_collection:Collection,num_returns):
        self.naics_collection = naics_collection
        self.parent_num_returns = num_returns
    
    def query_parent(self,query_embeddings:list[float],parent_num_results:int=5)->tuple[list[str],list[str]]:
        """Get relevant parent documents for a given question

        Args:
            query_embeddings (str): Question embeddings to get relevant parent documents for
            parent_num_results (int, optional): _description_. Defaults to 5.

        Returns:
            tuple[list[str],list[str]]: relevant parent NAICS IDs and industry names
        """
        relevant_parent_documents = self.naics_collection.query(
            query_embeddings = query_embeddings,
            n_results = parent_num_results,
            where = {"type":"PARENT"}
        )

        parent_naics_ids = [parent_metadata["NAICS CODE"] for parent_metadata in relevant_parent_documents["metadatas"][0]]
        parent_industry_names = [pd for pd in relevant_parent_documents["documents"][0]]
        
        return parent_naics_ids, parent_industry_names

class ChildQueryVectorDB(QueryVectorDatabase):
    """Query Vector Database for child results

    Args:
        naics_collection (Collection): ChromaDB Vector database collection
        num_returns (int): Number of relevant child documents to return
    """
    def __init__(self,naics_collection:Collection,num_returns):
        self.naics_collection = naics_collection
        self.child_num_returns = num_returns
    
    def query_parent(self,query_embeddings:list[list[float]])->tuple[list[str],list[str]]:
        """Get relevant parent documents for a given question

        Args:
            naics_collection (Collection): ChromaDB Vector database collection
            query_embeddings (str): Question embeddings to get relevant parent documents for
            parent_num_results (int, optional): _description_. Defaults to 5.

        Returns:
            tuple[list[str],list[str]]: relevant parent NAICS IDs and industry names
        """
        relevant_child_documents = self.naics_collection.query(
            query_embeddings = query_embeddings,
            n_results = self.child_num_returns,
            where = {"type":"PARENT"}
        )

        child_naics_ids = [parent_metadata["NAICS CODE"] for parent_metadata in relevant_child_documents["metadatas"][0]]
        child_industry_names = [pd for pd in relevant_child_documents["documents"][0]]
        
        return child_naics_ids, child_industry_names
