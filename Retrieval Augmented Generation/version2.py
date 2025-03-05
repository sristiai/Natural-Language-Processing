#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import pandas as pd
import nltk
import torch
import re
import logging
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from transformers import pipeline
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from langchain.schema import Document
from data_extraction import extract_data

# Download NLTK data if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_keywords_basic(text):
    """
        Extract keywords from a given text by removing stopwords and non-alphanumeric words.

        Parameters
        ----------
        text : str
            The input text from which keywords are to be extracted.

        Returns
        -------
        str
            A comma-separated string of keywords that are not in the list of stopwords and are alphanumeric.
    """

    # Define stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()  # Split the text into words based on spaces
    keywords = [word for word in words if word.lower() not in stop_words and word.isalnum()]
    return ', '.join(keywords)


def rerank_documents(docs):
    '''
    :param docs: List of docs copntaining the content to be re ranked
    :return:list of Document:
            A list of Document objects reranked by their relevance scores in descending order.
    '''
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

    inputs = tokenizer([doc.page_content for doc in docs], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    scores = outputs.logits[:, 1].numpy()
    ranked_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
    return ranked_docs


def autocut_documents(docs, query, model_name, threshold=0.0005):
    '''
    :param docs: A list of Document objects containing the content to be evaluated.
    :param query: The query text to evaluate the relevance of the documents against.
    :param model_name:  The name of the Hugging Face model to be used for relevance classification.
    :param threshold: The minimum relevance score for a document to be included in the filtered list. Defaults to 0.0005.

    :return:
    list of Document:
                A list of Document objects that meet the relevance threshold.

    1. Initializes a Hugging Face text-classification pipeline with the specified model and tokenizer.
        2. Iterates over the list of documents.
        3. For each document:
            - Combines the query and document content with a `[SEP]` separator.
            - Computes the relevance score using the classification model.
            - Includes the document in the output list if its score meets or exceeds the threshold.
        4. Returns the filtered list of documents.
    '''

    # Filter documents below a specified relevance threshold using a Hugging Face model.

    # Load pipeline from Hugging Face Hub
    # Sometimes we retrieve top five  then it might create noise for 4th or 5th
    # threshold is checked for  score and then passed it to llm.
    reranker = pipeline("text-classification", model=model_name, tokenizer=model_name)

    filtered_docs = []
    for doc in docs:
        # Combine query and document content with separator
        input_text = f"{query} [SEP] {doc.page_content}"

        # Predict relevance score
        result = reranker(input_text, truncation=True, max_length=512)
        score = result[0]["score"]  # Extract relevance score
        if score >= threshold:
            filtered_docs.append(doc)

    return filtered_docs


def main():
    """
        Main function for question-context extraction, keyword processing, and summarization.

        This function processes a dataset of questions and multiple-choice options, identifies relevant
        contexts for each option from a given set of source data, computes similarities using embeddings,
        and generates summarized answers for the most relevant context. The results are saved to an Excel file.

        Workflow:
        1. **Summarization Setup**: Initializes the summarizer pipeline using the T5-small model.
        2. **Data Loading and Preprocessing**:
            - Loads question and source data.
            - Normalizes and processes the options column into individual columns (A, B, C, D).
            - Extracts keywords for each option.
        3. **Embedding Model Setup**: Sets up a SentenceTransformer model (`all-MiniLM-L6-v2`) for embedding.
        4. **Iterative Processing**:
            - Iterates over questions in the dataset.
            - Extracts keywords for the question and options.
            - Filters relevant contexts from the source data based on keyword matching.
            - Computes similarity scores between the question and the filtered contexts using cosine similarity.
            - Retrieves the top-k relevant contexts.
        5. **Context Summarization**:
            - Reranks and refines relevant documents.
            - Summarizes the most relevant contexts using the T5 summarizer.
        6. **Output**: Updates the dataset with the extracted/summarized context and saves the results to an Excel file.

        Parameters:
            None

        Returns:
            None: The function directly processes data and writes the results to an Excel file.
        """
    summariser = pipeline("summarization", model="t5-small")

    df_ques, sources_df = extract_data()

    options_df = pd.json_normalize(df_ques['options'])
    df = pd.concat([df_ques.drop(columns=['options']), options_df], axis=1)

    for col in ['A', 'B', 'C', 'D']:
        df[col] = df[col].apply(extract_keywords_basic)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer('all-MiniLM-L6-v2', device=device) # Choose a model as per your requirement
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
            option = options[j]
            option_content = random_sample[option][i]

            keywords = extract_keywords_basic(option_content)
            keywords_list = keywords.split(', ')
            filtered_sources_df_1 = sources_df[
                sources_df['QUESTION'].str.contains('|'.join(keywords_list), case=False, na=False) |
                sources_df['ANSWER'].str.contains('|'.join(keywords_list), case=False, na=False)]

            filtered_sources_df = filtered_sources_df_1[
                filtered_sources_df_1['QUESTION'].str.contains('|'.join(ques_keywords_list), case=False, na=False) |
                filtered_sources_df_1['ANSWER'].str.contains('|'.join(ques_keywords_list), case=False, na=False)]
            logger.info(filtered_sources_df.shape)
            if filtered_sources_df.shape[0] > 10000:
                continue
            if filtered_sources_df.shape[0] == 0:
                continue

            question_answer_pairs = filtered_sources_df[['QUESTION', 'ANSWER']].apply(
                lambda row: (str(row['QUESTION']) if pd.notna(row['QUESTION']) else '') + " " + (
                    str(row['ANSWER']) if pd.notna(row['ANSWER']) else ''), axis=1).tolist()

            if filtered_sources_df.shape[0] == 1:
                answer = filtered_sources_df["ANSWER"].iloc[0]
                random_sample.loc[i, f"Context_{option}"] = answer
                continue
            chunk_size = 10
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
            scores = top_results[0].tolist()
            indices = top_results[1].tolist()
            context = []
            for k in range(top_k):
                idx = int(top_results[1][k])
                context.append(flat_list[idx])

            context1 = []
            for idx, score in zip(indices, scores):
                context1.append({"chunk": flat_list[idx], "similarity_score": score})
            for item in context1:
                print(f"Chunk: {item['chunk']}")
                print(f"Similarity Score: {item['similarity_score']}")
                print("------")

            # Convert list of texts into Document objects
            documents = [Document(page_content=text) for text in context]

            ranked = rerank_documents(documents)

            # Remove the irrelevant documents
            autocut_doc = autocut_documents(ranked, ques, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                                            threshold=0.00005)
            retrieved = [doc.page_content for doc in autocut_doc]
            extracted_text = retrieved[:min(1, len(retrieved))]
            extracted_text = " ".join(extracted_text)
            if len(extracted_text) > 500:
                answer = summariser(extracted_text, max_length=500, min_length=30, do_sample=False)
            else:
                answer = summariser(extracted_text, min_length=30, do_sample=False)

            random_sample.loc[i, f"Context_{option}"] = answer
            logger.info("Writing the output to the file")
            random_sample.to_excel(
                os.path.join("/home/bcae/Code_Repository/data", "version2.xlsx"))


if __name__ == "__main__":
    main()
