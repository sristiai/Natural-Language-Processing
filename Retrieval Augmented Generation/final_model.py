#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import logging
import nltk
import torch
import re
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from data_extraction import extract_data
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_keywords_basic(text):
    """
       Extracts keywords from a given text by removing stop words and non-alphanumeric tokens.

       Parameters
       ----------
       text : str
           The input text from which keywords are to be extracted.

       Returns
       -------
       str
           A comma-separated string of keywords extracted from the input text.
    """
    stop_words = set(stopwords.words('english'))
    words = text.split()  # Split the text into words based on spaces
    keywords = [word for word in words if word.lower() not in stop_words and word.isalnum()]
    return ', '.join(keywords)


class LangchainEmbeddingWrapper(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of documents."""
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query."""
        return self.model.encode(text, convert_to_numpy=True).tolist()


def get_vector_store(text_chunks):
    """
    Convert text chunks into vector embeddings and return a retriever with compression.

    Parameters
    ----------
    text_chunks : list of str
        A list of text chunks to be converted into vector embeddings.

    Returns
    -------
    ContextualCompressionRetriever or None
        A retriever with contextual compression applied, or `None` if `text_chunks` is empty.

    Notes
    -----
    - The function uses the "sentence-transformers/all-MiniLM-L6-v2" model for generating embeddings.
    - FAISS (Facebook AI Similarity Search) is used as the vector store for efficient similarity searches.
    - FlashrankRerank is applied as a compression mechanism to optimize the retrieval process.
    - The retriever is configured to retrieve the top 6 results (`k=6`).
    """
    if not text_chunks:
        return None

    # Define the embedding model to convert text to vectors
    embedding_model = LangchainEmbeddingWrapper("sentence-transformers/all-MiniLM-L6-v2")

    # Create a FAISS vector store from the text chunks and their embeddings
    vector_store = FAISS.from_texts(text_chunks, embedding=embedding_model)

    # Define the retriever using the FAISS vector store with a search limit of k=6
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    FlashrankRerank.model_rebuild()
    # Apply compression using FlashrankRerank
    compressor = FlashrankRerank()

    # Return a ContextualCompressionRetriever that uses the base compressor and retriever
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)


def main():
    """
        Processes and analyzes a dataset of questions and their associated options, using NLP techniques
        to extract keywords, compute semantic similarity, and summarize context for each option.

        This function performs the following tasks:
        1. Loads and preprocesses question and option data.
        2. Applies keyword extraction to options.
        3. Encodes questions and options into embeddings for semantic similarity computation.
        4. Filters and summarizes relevant context based on similarity.
        5. Saves the processed data to an Excel file.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The function saves the output data, enriched with summarized context, to an Excel file.
"""

    summariser = pipeline("summarization", model="t5-small", device=0)

    df_ques, sources_df = extract_data()

    options_df = pd.json_normalize(df_ques['options'])
    df = pd.concat([df_ques.drop(columns=['options']), options_df], axis=1)

    # Download NLTK data if not already downloaded
    nltk.download('punkt')
    nltk.download('stopwords')

    # Apply the function to each of the option columns
    for col in ['A', 'B', 'C', 'D']:
        df[col] = df[col].apply(extract_keywords_basic)

    # Load a pre-trained model

    model = SentenceTransformer("abhinand/MedEmbed-small-v0.1")

    random_sample = pd.read_excel("/home/bcae/Code_Repository/data/TESTING_QUESTIONS.xlsx")

    options = ['A', 'B', 'C', 'D']

    for i in range(random_sample.shape[0]):
        logger.info(i)
        ques = random_sample["question"][i]
        ques_keywords = extract_keywords_basic(ques)
        ques_keywords_list = ques_keywords.split(', ')
        question_embedding = model.encode(ques)
        for j in range(len(options)):
            logger.info(options)
            option = options[j]
            option_content = random_sample[option][i]
            keywords = extract_keywords_basic(option_content)
            keywords_list = keywords.split(', ')
            '''
                Initial Filtering:Keywords from the options are matched 
                against rows in the source dataset.This initial pass significantly 
                 reduces the dataset size by filtering out irrelevant rows.
                Secondary Filtering:
                The filter was applied with respect to the keywords in 
                option to rather narrow down the relevant text. This helped in 
                the cases where options were very common words like “high”, “low” or numbers.
            '''
            filtered_sources_df_1 = sources_df[
                sources_df['QUESTION'].
                str.contains('|'.join(keywords_list), case=False, na=False) |
                sources_df['ANSWER'].
                str.contains('|'.join(keywords_list), case=False, na=False)]

            filtered_sources_df = filtered_sources_df_1[
                filtered_sources_df_1['QUESTION'].
                str.contains('|'.join(ques_keywords_list), case=False, na=False) |
                filtered_sources_df_1['ANSWER'].
                str.contains('|'.join(ques_keywords_list), case=False, na=False)]
            logger.info("Shape of the filtered dataframe", filtered_sources_df.shape)

            if filtered_sources_df.empty:
                logger.info(f"No filtered sources found for question {ques}. Skipping...")
                continue

            # question_answer_pairs = filtered_sources_df[['QUESTION', 'ANSWER']].apply(lambda row: row['QUESTION'] + " " + row['ANSWER'], axis=1).tolist()
            question_answer_pairs = filtered_sources_df[['QUESTION', 'ANSWER']].apply(
                lambda row: (str(row['QUESTION']) if pd.notna(row['QUESTION']) else '') + " " + (
                    str(row['ANSWER']) if pd.notna(row['ANSWER']) else ''), axis=1).tolist()

            if filtered_sources_df.shape[0] == 1:
                answer = filtered_sources_df["ANSWER"].iloc[0]
                random_sample.loc[i, f"Context_{option}"] = answer
                continue

            chunk_size = 10
            # question answer paired chunking
            chunks = [question_answer_pairs[i:i + chunk_size] for i in range(0, len(question_answer_pairs), chunk_size)]

            # Flatten the chunks into a single list of question-answer pairs
            flat_list = [item for sublist in chunks for item in sublist]

            # Displaying the flattened list
            embeddings = model.encode(flat_list)

            # Step 4: Compute similarity between the question and each chunk
            similarities = util.cos_sim(question_embedding, embeddings)

            # Step 5: Find the most similar chunks
            top_k = min(similarities.shape[1], 5)  # Number of top relevant chunks to retrieve
            top_results = torch.topk(similarities[0], k=top_k)
            context = []
            for score, idx in zip(top_results[0], top_results[1]):
                idx = int(idx)  # Convert tensor to integer

                context.append(flat_list[idx])

            # Using compression to get the relevant documents
            compression_retriever = get_vector_store(context)
            compressed_docs = compression_retriever.get_relevant_documents(ques)
            retrieved = [doc.page_content for doc in compressed_docs]
            final_context = " ".join(retrieved)
            # Summarising if length exceeded
            # we can also summarise without checking the condition
            if len(final_context) > 250:
                answer = summariser(final_context, max_length=250, min_length=100, do_sample=False)
            else:
                answer = summariser(final_context, min_length=100, do_sample=False)

            random_sample.loc[i, f"Context_{option}"] = answer
            logger.info("Writing the output to the file")
            '''
            Saves the context for each question's options
            '''
            random_sample.to_excel(
                os.path.join("/home/bcae/NLP/extracted/RND_Extraction/output", "Final_model.xlsx"))

if __name__ == "__main__":
    main()
