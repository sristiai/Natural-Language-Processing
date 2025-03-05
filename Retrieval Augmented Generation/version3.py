#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import torch
import re
import nltk
import logging
from data_extraction import extract_data
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_keywords_basic(text):
    """
        Extract keywords from a given text by removing stopwords and filtering alphanumeric words.

        Parameters
        ----------
        text : str
            The input text from which keywords are to be extracted.

        Returns
        -------
        str
            A comma-separated string of keywords. Keywords are words that:
            - Are not in the stopwords list.
            - Are alphanumeric.
    """
    stop_words = set(stopwords.words('english'))
    words = text.split()  # Split the text into words based on spaces
    keywords = [word for word in words if word.lower() not in stop_words and word.isalnum()]
    return ', '.join(keywords)


def main():
    """
        Process a dataset of questions and multiple-choice options to extract and match relevant contexts.

        This function performs the following tasks:
        - Preprocesses the input question and options dataset.
        - Extracts keywords for each question and option using a basic keyword extraction method.
        - Uses a pre-trained SentenceTransformer model to compute embeddings.
        - Matches relevant contexts from a source dataset based on keyword overlap and similarity scores.
        - Stores the most relevant context for each question-option pair in the dataset.
        - Saves the processed dataset to an Excel file.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The function processes the input data and writes the output to an Excel file.
        """
    df_ques, sources_df = extract_data()

    options_df = pd.json_normalize(df_ques['options'])
    df = pd.concat([df_ques.drop(columns=['options']), options_df], axis=1)

    # Download NLTK data if not already downloaded
    nltk.download('punkt')
    nltk.download('stopwords')

    # Define stopwords

    for col in ['A', 'B', 'C', 'D']:
        df[col] = df[col].apply(extract_keywords_basic)
    # Load a pre-trained model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)  # Choose a model as per your requirement

    '''
        This is done to test for 10 examples. or else we can just pass orignal dataframe here
        or we can randomly select using this code :
        random_sample = df.sample(n=10, random_state=42)
    '''

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
                sources_df['QUESTION'].str.contains('|'.join(keywords_list), case=False, na=False) |
                sources_df['ANSWER'].str.contains('|'.join(keywords_list), case=False, na=False)]

            filtered_sources_df = filtered_sources_df_1[
                filtered_sources_df_1['QUESTION'].str.contains('|'.join(ques_keywords_list),
                                                               case=False, na=False) |
                filtered_sources_df_1['ANSWER'].str.contains('|'.join(ques_keywords_list),
                                                             case=False, na=False)]

            if filtered_sources_df.shape[0] > 10000:
                continue

            if filtered_sources_df.empty:
                print(f"No filtered sources found for question {ques}. Skipping...")
                continue

            question_answer_pairs = filtered_sources_df[['QUESTION', 'ANSWER']].apply(
                lambda row: (str(row['QUESTION']) if pd.notna(row['QUESTION']) else '') + " " + (
                    str(row['ANSWER']) if pd.notna(row['ANSWER']) else ''), axis=1).tolist()

            if filtered_sources_df.shape[0] == 1:
                answer = filtered_sources_df["ANSWER"].iloc[0]
                random_sample.loc[i, f"Context_{option}"] = answer
                continue
            chunk_size = 10
            chunks = [question_answer_pairs[i:i + chunk_size]
                      for i in range(0, len(question_answer_pairs), chunk_size)]

            # Flatten the chunks into a single list of question-answer pairs
            flat_list = [item for sublist in chunks for item in sublist]

            # Displaying the flattened list
            embeddings = model.encode(flat_list)

            # Step 4: Compute similarity between the question and each chunk
            similarities = util.cos_sim(question_embedding, embeddings)

            # Step 5: Find the most similar chunks
            top_k = min(similarities.shape[1], 3)  # Number of top relevant chunks to retrieve
            top_results = torch.topk(similarities[0], k=top_k)

            max_index = torch.argmax(top_results[0]).item()  # Get the index of the max value
            max_score_idx = int(top_results[1][max_index])
            answer = flat_list[max_score_idx]

            random_sample.loc[i, f"Context_{option}"] = answer
            logger.info("Writing the output to the file")
            random_sample.to_excel(
                os.path.join("/home/bcae/Code_Repository/data", "version3.xlsx"))

if __name__ == "__main__":
    main()
