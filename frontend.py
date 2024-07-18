import streamlit as st
import re
from annotated_text import annotated_text, annotation
from app.prediction import Prediction
from app.vectordb import load_database
from typing import List, Union, Tuple

st.title("NAICS Classification")

@st.cache_resource
def load_resources():
    naics_collection = load_database()
    prediction = Prediction(naics_collection)
    return prediction

prediction = load_resources()

def process_question(mod_question:str)->List[Union[str,Tuple[str]]]:
    """Get annotated words for streamlit view

    Args:
        mod_question (str): Question to be modified

    Returns:
        _type_: _description_
    """
    for spans, inames in zip(ner_entities_span,industry_names_code):
        curr_end = spans[1]
        ner = f"#-{inames.industry}:{inames.naics_code}-# "
        mod_question = mod_question[:curr_end] + ner + mod_question[curr_end+1:]
    pattern = r'(#[^#]*#)|\s+'

    # Splitting the sentence based on the pattern
    chunks = [chunk for chunk in re.split(pattern, mod_question) if chunk and not chunk.isspace()]
    annotated_words = []

    for chunk in chunks:
        if "#-" in chunk and "-#" in chunk:
            annotated_words[-1] = (annotated_words[-1],chunk[2:-2])
        else:
            annotated_words.append(chunk+" ")
    return annotated_words

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "NAICS Classification Agent"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question := st.chat_input():
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Answering..."):
            ner_entities_span, industry_names_code = prediction.pipeline(question)
            # response = f"Entities: {ner_entities_span}\nIndustry Names: {industry_names_code}"
            annotated_words = process_question(question)
            st.write(annotated_text(annotated_words))
    message = {"role": "assistant", "content": annotated_text(annotated_words)}
    st.session_state.messages.append(message)