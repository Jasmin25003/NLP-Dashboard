# NLP Analysis Dashboard

## Problem Statement 5

Prepare a dashboard to provide an interface for the user to perform the following NLP operations:

1. Named-Entity Recognition
2. POS Tagging
3. POS Distribution
4. Lemmatization
5. Stemming
6. Morphology
7. Dependencies - style=dep

## Project Description

This project is an interactive Natural Language Processing dashboard developed using Python and Streamlit.

The dashboard allows the user to enter text and perform different NLP operations through a simple interface. The text is processed using spaCy and NLTK, while Pandas and Matplotlib are used for displaying results and visualizations.

## Technologies Used

- Python
- Streamlit
- spaCy
- NLTK
- Pandas
- Matplotlib

## Features

### 1. Named-Entity Recognition

Identifies important entities present in the input text, such as:

- PERSON
- ORGANIZATION
- LOCATION
- DATE

The project also provides entity visualization.

### 2. POS Tagging

Identifies the Part-of-Speech of each word in the input text.

Examples include:

- Noun
- Verb
- Adjective
- Adverb
- Pronoun
- Determiner

### 3. POS Distribution

Displays the frequency of different Part-of-Speech tags using a bar chart.

### 4. Lemmatization

Converts words into their base or dictionary form.

Example:

```text
planning → plan
announced → announce