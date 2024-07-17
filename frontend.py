import streamlit as st
from annotated_text import annotated_text, annotation
from app.prediction import Prediction
from app.vectordb import load_database

st.title("NAICS Classification")

@st.cache_resource
def load_resources():
    naics_collection = load_database()
    prediction = Prediction(naics_collection)
    return prediction

prediction = load_resources()

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I help you?"}]

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
            annotated_words = [f"{qs} " for qs in question.split(" ")]
            for spans, inames in zip(ner_entities_span,industry_names_code):
                curr_start = spans[0]
                curr_end = spans[1]
                # get the NER words
                ner = question[curr_start:curr_end].split(" ")
                for aw in annotated_words:
                    if aw.strip() in ner:
                        aw = (aw,f"{inames.industry}: {inames.naics_code}")
            response = annotated_text(tuple(annotated_words))
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)