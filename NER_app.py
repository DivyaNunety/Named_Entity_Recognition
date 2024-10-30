import streamlit as st
import spacy_streamlit as spt
import spacy
import networkx as nx
import matplotlib.pyplot as plt

# Load the spaCy NLP model
nlp = spacy.load("en_core_web_sm")

def generate_knowledge_graph(raw_text):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Process the text with spaCy
    doc = nlp(raw_text)

    # Create a directed graph
    G = nx.DiGraph()

    # Extract named entities
    entities = list(doc.ents)

    # Add nodes for named entities
    for ent in entities:
        G.add_node(ent.text, type=ent.label_)

    # Connect named entities in the order they appear
    for i in range(len(entities) - 1):
        prev_ent = entities[i]
        next_ent = entities[i + 1]
        G.add_edge(prev_ent.text, next_ent.text)

    # Draw the graph
    plt.figure(figsize=(12, 8))
    pos = nx.kamada_kawai_layout(G)
    nx.draw(
        G, pos, with_labels=True, node_size=1000, font_size=8, font_weight='bold',
        node_color='skyblue', edge_color='gray'
    )
    plt.title("Knowledge Graph from Named Entities in Text")
    st.pyplot()

def main():
    st.title('Named Entity Recognition (NER) App')
    menu = ['Tokenize', 'NER', 'Knowledge Graph']
    choice = st.sidebar.selectbox('Menu', menu)
    raw_text = st.text_area('Text To Tokenize', '')

    if choice == 'Tokenize':
        st.subheader('Word Tokenization')
  #      raw_text = st.text_area('Text To Tokenize', '')
        docs = nlp (raw_text)
        if st.button('Tokenize'):
            spt.visualize_tokens (docs)

    elif choice == 'NER':
        st.subheader('Named Entity Recognition')
        docs = nlp (raw_text)
        spt.visualize_ner(docs)

    elif choice == 'Knowledge Graph':
        st.subheader('Knowledge Graph Generation')
       # raw_text = st.text_area('Text To Generate Knowledge Graph', '')
       # if st.button('Generate Knowledge Graph'):
        generate_knowledge_graph(raw_text)

if __name__ == "__main__":
    main()
