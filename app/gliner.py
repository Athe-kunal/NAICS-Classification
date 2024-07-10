from gliner import GLiNER
import yaml
import torch
import logging

logging.basicConfig(filename="gliner.log",
                    format='%(asctime)s %(message)s',
                    filemode='a')

with open("config.yaml") as stream:
    try:
        config_params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

def load_model():
    model = GLiNER.from_pretrained(config_params['GLINER']['MODEL_NAME'])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    return model

# global gliner_model 
# gliner_model = load_model()


def get_entities(gliner_model,question:str,threshold:float=0.2):
    """Get the entities of a given question from the GLI-NER model based on the given threshold

    Args:
        gliner_model (GLiNER): The GLI-NER model
        question (str): question to get the entities
        threshold (float, optional): threshold for considering a phrase as NER. Defaults to 0.2.
    """
    entities = gliner_model.predict_entities(question, config_params['GLINER']['MODEL_NAME'], threshold=threshold)
    logging.info(f"Question: {question}\n Entities: {entities}")
    return entities