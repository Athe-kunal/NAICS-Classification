import json
import chromadb.utils
import chromadb.utils.embedding_functions
import yaml
import chromadb
import torch
from dotenv import load_dotenv, find_dotenv
import os
from typing import Union
from chromadb.utils.embedding_functions import (
    OpenAIEmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
)
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv(), override=True)

with open("config.yaml") as stream:
    try:
        config_params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


def _get_docs_metadata() -> tuple[list[str], list[str]]:
    """Returns the documents and metadata to be stored in the vector database for all the NAICS industries
        It has two kind of documents
            PARENTS: It has the parent code and parent_industry description
            CHILD: It has the child code, child_industry description and it's corresponding parent industry name

    Returns:
        tuple[list[str],list[str]]: The documents and their corresponding metadata
    """
    with open(config_params["NAICS"]["SAVE_PATH"], "r") as openfile:
        naics_ids = json.load(openfile)
    docs = []
    metadata = []

    for nids in naics_ids:
        parent_name = nids["parent_industry_name"]
        docs.append(nids["parent_industry_desc"])
        metadata.append(
            {
                "NAICS CODE": nids["parent_code"],
                "PARENT NAME": parent_name,
                "DESCRIPTION": nids["parent_industry_desc"],
                "TYPE": "PARENT",
            }
        )

        for child_naics in nids["child_naics_dict"]:
            docs.append(child_naics["child_description"])
            metadata.append(
                {
                    "NAICS CODE": ", ".join(child_naics["child_code"]),
                    "DESCRIPTION": child_naics["child_description"],
                    "CHILD NAME": child_naics["child_industry_name"],
                    "TYPE": "CHILD",
                    "PARENT": parent_name,
                }
            )
    return docs, metadata


def get_embedding_fn() -> (
    Union[SentenceTransformerEmbeddingFunction, OpenAIEmbeddingFunction]
):
    """Returns the embedding function based on the configuration parameters

    Returns:
        chromadb.utils.embedding_functions.EmbeddingFunction: The embedding function to be used
    """
    embedding_model_type = config_params["VECTORDB"]["EMBEDDING_MODEL_TYPE"]

    assert embedding_model_type in [
        "sentence_transformer",
        "openai",
    ], "Only sentence_transformer and openai are supported"
    if embedding_model_type == "sentence_transformer":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return SentenceTransformerEmbeddingFunction(
            model_name=config_params["VECTORDB"]["EMBEDDING_MODEL_NAME"], device=device
        )
    elif embedding_model_type == "openai":
        _ = load_dotenv(find_dotenv(), override=True)
        return OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name=config_params["VECTORDB"]["EMBEDDING_MODEL_NAME"],
        )


def build_database():
    naics_docs, naics_metadata = _get_docs_metadata()

    client = chromadb.PersistentClient(path=config_params["VECTORDB"]["DATABASE_PATH"])
    emb_fn = get_embedding_fn()
    naics_collection = client.create_collection(
        name=config_params["VECTORDB"]["COLLECTION_NAME"]
        + "-"
        + config_params["VECTORDB"]["EMBEDDING_MODEL_TYPE"],
        embedding_function=emb_fn,
    )
    naics_collection.add(
        documents=naics_docs,
        metadatas=naics_metadata,
        ids=[str(i) for i in range(len(naics_docs))],
    )
    print("Database built successfully.")
    return naics_collection


def load_database():
    client = chromadb.PersistentClient(path=config_params["VECTORDB"]["DATABASE_PATH"])
    emb_fn = get_embedding_fn()
    return client.get_collection(
        name=config_params["VECTORDB"]["COLLECTION_NAME"]
        + "-"
        + config_params["VECTORDB"]["EMBEDDING_MODEL_TYPE"],
        embedding_function=emb_fn,
    )
