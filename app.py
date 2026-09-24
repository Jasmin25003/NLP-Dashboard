import streamlit as st
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from spacy import displacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Stemmer
stemmer = PorterStemmer()

# Page configuration
st.set_page_config(
    page_title="NLP Dashboard",
    page_icon="🔤",
    layout="wide"
)

# Title
st.title("🔤 NLP Analysis Dashboard")
st.write("Perform different Natural Language Processing operations on your text.")

# Input box
text = st.text_area(
    "Enter / Input your text:",
    height=150,
    placeholder="Enter a sentence or paragraph here..."
)

if text.strip():

    doc = nlp(text)

    # Sidebar
    st.sidebar.header("NLP Operations")

    operation = st.sidebar.selectbox(
        "Select Operation",
        [
            "Dashboard",
            "Named-Entity Recognition",
            "POS Tagging",
            "POS Distribution",
            "Lemmatization",
            "Stemming",
            "Morphology",
            "Dependencies"
        ]
    )

    # ------------------------------------------------
    # Dashboard
    # ------------------------------------------------
    if operation == "Dashboard":

        st.subheader("📊 Text Statistics")

        col1, col2, col3, col4 = st.columns(4)

        tokens = [token for token in doc]
        entities = list(doc.ents)
        punctuation = [token for token in doc if token.is_punct]

        col1.metric("Tokens", len(tokens))
        col2.metric("Entities", len(entities))
        col3.metric("Punctuation", len(punctuation))
        col4.metric("Sentences", len(list(doc.sents)))

        st.subheader("Processed Text")

        data = []

        for token in doc:
            data.append({
                "Token": token.text,
                "POS": token.pos_,
                "Lemma": token.lemma_,
                "Dependency": token.dep_
            })

        st.dataframe(pd.DataFrame(data), use_container_width=True)

    # ------------------------------------------------
    # Named Entity Recognition
    # ------------------------------------------------
    elif operation == "Named-Entity Recognition":

        st.subheader("🏷 Named-Entity Recognition")

        if doc.ents:
            entity_data = []

            for ent in doc.ents:
                entity_data.append({
                    "Entity": ent.text,
                    "Label": ent.label_,
                    "Description": spacy.explain(ent.label_)
                })

            st.dataframe(
                pd.DataFrame(entity_data),
                use_container_width=True
            )

            st.subheader("Entity Visualization")

            html = displacy.render(
                doc,
                style="ent",
                jupyter=False
            )

            st.components.v1.html(html, height=250)

        else:
            st.info("No named entities found.")

    # ------------------------------------------------
    # POS Tagging
    # ------------------------------------------------
    elif operation == "POS Tagging":

        st.subheader("🏷 Part-of-Speech Tagging")

        pos_data = []

        for token in doc:
            pos_data.append({
                "Token": token.text,
                "POS": token.pos_,
                "Tag": token.tag_,
                "Description": spacy.explain(token.tag_)
            })

        st.dataframe(
            pd.DataFrame(pos_data),
            use_container_width=True
        )

    # ------------------------------------------------
    # POS Distribution
    # ------------------------------------------------
    elif operation == "POS Distribution":

        st.subheader("📈 POS Distribution")

        pos_counts = {}

        for token in doc:
            if not token.is_punct:
                pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1

        df = pd.DataFrame(
            list(pos_counts.items()),
            columns=["POS", "Count"]
        )

        st.dataframe(df, use_container_width=True)

        fig, ax = plt.subplots()

        ax.bar(df["POS"], df["Count"])
        ax.set_xlabel("Part of Speech")
        ax.set_ylabel("Frequency")
        ax.set_title("POS Distribution")

        st.pyplot(fig)

    # ------------------------------------------------
    # Lemmatization
    # ------------------------------------------------
    elif operation == "Lemmatization":

        st.subheader("🔤 Lemmatization")

        lemma_data = []

        for token in doc:
            if not token.is_punct:
                lemma_data.append({
                    "Word": token.text,
                    "Lemma": token.lemma_
                })

        st.dataframe(
            pd.DataFrame(lemma_data),
            use_container_width=True
        )

    # ------------------------------------------------
    # Stemming
    # ------------------------------------------------
    elif operation == "Stemming":

        st.subheader("🌱 Stemming")

        stem_data = []

        for token in doc:
            if not token.is_punct:
                stem_data.append({
                    "Word": token.text,
                    "Stem": stemmer.stem(token.text)
                })

        st.dataframe(
            pd.DataFrame(stem_data),
            use_container_width=True
        )

    # ------------------------------------------------
    # Morphology
    # ------------------------------------------------
    elif operation == "Morphology":

        st.subheader("🔬 Morphological Analysis")

        morphology_data = []

        for token in doc:
            if not token.is_punct:
                morphology_data.append({
                    "Word": token.text,
                    "POS": token.pos_,
                    "Morphology": str(token.morph),
                    "Features": token.morph.to_dict()
                })

        st.dataframe(
            pd.DataFrame(morphology_data),
            use_container_width=True
        )

    # ------------------------------------------------
    # Dependencies
    # ------------------------------------------------
    elif operation == "Dependencies":

        st.subheader("🔗 Dependency Parsing")

        st.write(
            "Dependency tree showing the grammatical relationship "
            "between words."
        )

        html = displacy.render(
            doc,
            style="dep",
            options={
                "compact": True,
                "distance": 100
            },
            jupyter=False
        )

        st.components.v1.html(
            html,
            height=500,
            scrolling=True
        )

else:
    st.info("👆 Enter some text above to start NLP analysis.")