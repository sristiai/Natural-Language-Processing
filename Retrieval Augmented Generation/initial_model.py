#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import nltk
import torch
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from data_extraction import extract_data
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import PromptTemplate
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nltk.download('punkt')
nltk.download('stopwords')


def extract_keywords_basic(text):
    """
        Extract keywords from a given text by removing stop words and non-alphanumeric words.

        Parameters:
            text (str): The input text from which to extract keywords.

        Returns:
            str: A string of keywords separated by commas.

        Note:
            - This function assumes `stop_words` is a predefined list or set of stop words.
            - Words are considered keywords if they are alphanumeric and not in the `stop_words`.
    """
    stop_words = set(stopwords.words('english'))
    words = text.split()  # Split the text into words based on spaces
    keywords = [word for word in words if word.lower() not in stop_words and word.isalnum()]
    return ', '.join(keywords)


def setup_prompt_template():
    template = """You are an expert in clinical medicine, specializing in patient case studies and treatment scenarios. You are presented with a question based on a detailed patient case study, which outlines the patient's condition, history, symptoms, and other relevant clinical details. 

Your task is to provide a **general, factual explanation** of one specific option from a multiple-choice question related to the case study. The explanation should provide an overview of the characteristics, mechanisms, or clinical implications of the option, but should not suggest or confirm whether it is the correct answer.

### Key Guidelines:
1. Focus on explaining the **relevance** and **context** of the **provided option** in relation to the patient's case.
2. Provide **objective** and **informative** details about the option, such as its role, significance, and clinical use, without making any judgment or drawing conclusions about the correct answer.
3. Limit your explanation to **under 500 words** while maintaining clarity and completeness.
4. Avoid defending the option as the correct answer. The goal is to present each option's explanation so that the user can understand its relevance and make their own informed decision.
5. Stricty Do not provide the answer
### Context for the Option:
{context}

### Patient Case Study Question:
{question}

### Informative Explanation (under 500 words):
"""
    return PromptTemplate(input_variables=["context", "question"], template=template)


def extract_informative_explanation(text, heading="Informative Explanation (under 500 words)", word_limit=500):
    '''

    :param text: The full text from which the section needs to be extracted.
    :param heading: The heading of the section to extract. Defaults to "Informative Explanation (under 500 words)".
    :param word_limit: The maximum number of words to include in the extracted section. Defaults to 500
    :return: The trimmed section of text under the specified heading.
    If the heading is not found, returns "Section not found."
    '''
    # Use a regex to find the specific heading and its content
    match = re.search(rf"{re.escape(heading)}:(.*?)(?:###|$)", text, re.S)
    if not match:
        return "Section not found."

    section_text = match.group(1).strip()

    # Limit the section to the specified number of words
    words = section_text.split()
    trimmed_text = ' '.join(words[:word_limit])
    return trimmed_text


def main():
    """
       Main function to process question-answer pairs, generate embeddings,
        perform similarity search, and utilize a
       language model to generate informative explanations.

       Workflow:
           1. Load and process data:
               - Extract data from pre-defined sources.
               - Normalize question options.
               - Perform chunking 1 question answer pair will be 1 chunk

           2. Generate and load embeddings:
               - Use SentenceTransformer to create embeddings for question-answer pairs.
               - Save embeddings to a file for future use, or load if already available.

           3. Process new questions:
               - For each question and its options, generate embeddings.
               - Perform similarity search using cosine similarity with pre-computed embeddings.
               - Retrieve the most relevant chunks of context.

           4. Use a language model (Gemma) to generate responses:
               - Format the prompt using a predefined template.
               - Pass the prompt to the language model for response generation.
               - Extract the informative explanation from the response.

           5. Save the results:
               - Save the generated responses and context for each option back to an Excel file.
       Returns:
           None. Saves the generated responses and contexts to an output Excel file.
       """
    '''
    References: https: // attri.ai / blog / retrieval - augmented - generation - rag - architecture
                https: // chatgpt.com
    '''
    current_dir = os.getcwd()
    base_dir = os.path.abspath(current_dir)
    embedding_file_path = os.path.join(base_dir, "Embedding", "VECTOR DATABASE", "embeddings.npy")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)  # Choose a model as per your requirement
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    pipe = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", device_map="auto")
    df_ques, sources_df = extract_data()
    options_df = pd.json_normalize(df_ques['options'])
    df = pd.concat([df_ques.drop(columns=['options']), options_df], axis=1)

    for col in ['A', 'B', 'C', 'D']:
        df[col] = df[col].apply(extract_keywords_basic)

    question_answer_pairs = sources_df[['QUESTION', 'ANSWER']].apply(
        lambda row: str(row['QUESTION']) + " " + str(row['ANSWER']), axis=1
    ).tolist()

    chunk_size = 1
    chunks = [question_answer_pairs[i:i + chunk_size] for i in range(0, len(question_answer_pairs), chunk_size)]

    # Flatten the chunks into a single list of question-answer pairs
    flat_list = [item for sublist in chunks for item in sublist]


    os.makedirs(os.path.dirname(embedding_file_path), exist_ok=True)

    if os.path.exists(embedding_file_path):
        # Load the embeddings from the file
        embeddings = np.load(embedding_file_path)
        logger.info("Embeddings loaded successfully.")
    else:
        # If the file doesn't exist, generate the embeddings
        # Assuming `flat_list` is already defined and contains the question-answer pairs
        embeddings = model.encode(flat_list)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(embedding_file_path), exist_ok=True)

        # Save the embeddings to the file
        np.save(embedding_file_path, embeddings)
        logger.info("Embeddings generated and saved successfully.")
    '''
    This is done to test for 10 examples. or else we can just pass orignal dataframe here
    or we can randomly select using this code :
    random_sample = df.sample(n=10, random_state=42)
    '''

    random_sample = pd.read_excel("/home/bcae/Code_Repository/data/TESTING_QUESTIONS.xlsx")

    options = ['A', 'B', 'C', 'D']
    '''
    In this loop we select each question and its options,
    create the embedding do the similarity search take the topK results
    and pass it LLM model and (Gemma) and then get trimmed result to not exceed maximum token limit
    '''
    for i in range(random_sample.shape[0]):
        ques = random_sample["question"][i]
        for j in range(len(options)):
            option = options[j]
            option_content = random_sample[option][i]
            question = ques + "option" + option
            question_embedding = model.encode(
                ques + "What is the relevance of the question with respect to option: " + option_content)

            if os.path.exists(embedding_file_path):
                # Load the embeddings from the file
                embeddings = np.load(embedding_file_path)
                logger.info("Embeddings loaded successfully.")
            else:
                ''' 
                If the file doesn't exist, generate the embeddings
                Assuming `flat_list` is already defined and contains the question-answer pairs
                '''
                embeddings = model.encode(flat_list)

                # Ensure the directory exists
                os.makedirs(os.path.dirname(embedding_file_path), exist_ok=True)

                # Save the embeddings to the file
                np.save(embedding_file_path, embeddings)
                logger.info("Embeddings generated and saved successfully.")

            # Step 4: Compute similarity between the question and each chunk
            similarities = util.cos_sim(question_embedding, embeddings)

            # Step 5: Find the most similar chunks
            top_k = min(similarities.shape[1], 5)  # Number of top relevant chunks to retrieve
            top_results = torch.topk(similarities[0], k=top_k)
            context = ""
            for score, idx in zip(top_results[0], top_results[1]):
                idx = int(idx)  # Convert tensor to integer
                context = context + flat_list[idx]

            prompt_template = setup_prompt_template()
            prompt = prompt_template.format(context=context, question=question)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            outputs = pipe.generate(input_ids, max_new_tokens=256)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Now you can use the generated prompt to get the answer
            answer = extract_informative_explanation(response)

            random_sample.loc[i, f"Context_{option}"] = answer
            '''
            Saves the context for each question's options
            '''
            logger.info("Writing the output to the file")
            random_sample.to_excel(
                os.path.join("/home/bcae/Code_Repository/data", "initial_model.xlsx"))


if __name__ == "__main__":
    main()
