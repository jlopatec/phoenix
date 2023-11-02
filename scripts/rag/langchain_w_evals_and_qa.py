"""
Llama Index implementation of a chunking and query testing system
"""
import datetime
import os
import pickle
import time
import traceback

from llama_index import download_loader
from sklearn.metrics import ndcg_score


import config
import numpy as np
import openai
import pandas as pd
import phoenix.experimental.evals.templates.default_templates as templates
import requests
from bs4 import BeautifulSoup
from phoenix.experimental.evals import (
    RAG_RELEVANCY_PROMPT_TEMPLATE_STR,
    OpenAIModel,
    llm_eval_binary,
    run_relevance_eval,
)
from plotresults import (
    plot_latency_graphs,
    plot_mean_average_precision_graphs,
    plot_mean_precision_graphs,
    plot_mrr_graphs,
    plot_ndcg_graphs,
    plot_percentage_incorrect,
)
from llama_index_w_eavls_and_qa import get_urls, read_strings_from_csv
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import TokenTextSplitter


class Doc:
    def __init__(self, id_, embedding, metadata, excluded_embed_metadata_keys, excluded_llm_metadata_keys,
                 relationships, hash, text, start_char_idx, end_char_idx, text_template,
                 metadata_template, metadata_seperator, page_content):
        self.id_ = id_
        self.embedding = embedding
        self.metadata = metadata
        self.excluded_embed_metadata_keys = excluded_embed_metadata_keys
        self.excluded_llm_metadata_keys = excluded_llm_metadata_keys
        self.relationships = relationships
        self.hash = hash
        self.text = text
        self.start_char_idx = start_char_idx
        self.end_char_idx = end_char_idx
        self.text_template = text_template
        self.metadata_template = metadata_template
        self.metadata_seperator = metadata_seperator
        self.page_content = page_content


def get_document(url):
    """
    Fetch the content of a document/webpage given its URL.

    Args:
    - url (str): The URL of the document/webpage.

    Returns:
    - str: The content of the document.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful

        # Parse the content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Here, we'll assume that the main content of the document is within <body> tag.
        document_content = soup.body.get_text(separator=' ', strip=True)

        return document_content

    except requests.RequestException as e:
        print(f"Error fetching document from URL {url}. Error: {e}")
        return None


def objectify_documents_chroma(documents):
    modified_docs = []
    for doc in documents:
        # Convert the document to a dictionary if it's not already one
        if isinstance(doc, dict):
            doc_dict = doc
        else:
            doc_dict = doc.__dict__

        # Add the new "page_content" key
        doc_dict["page_content"] = doc_dict["text"]
        modified_docs.append(doc_dict)

    max_doc_length = 1500  # set a reasonable value based on your needs and the model's limits
    transformed_docs = [{**doc, "page_content": doc["page_content"][:max_doc_length]} for doc in modified_docs]
    objectified_docs = [Doc(**doc) for doc in transformed_docs]
    return objectified_docs


def load_documents(base_url, save_base, file_name):
    """
    Load documents either from URLs or from a saved file.

    Args:
    - base_url (str): The base URL from which to fetch document URLs.
    - save_base (str): The path where the documents are saved/loaded.
    - file_name (str): The name of the pickle file where documents are saved/loaded.

    Returns:
    - list: A list of documents.
    """
    name = "BeautifulSoupWebReader"
    BeautifulSoupWebReader = download_loader(name)
    os.makedirs(os.path.dirname(save_base), exist_ok=True)
    if not os.path.exists(save_base + file_name):
        print(f"'{save_base}{file_name}' does not exist.")
        urls = get_urls(base_url)
        print(f"LOADED {len(urls)} URLS")

    print("GRABBING DOCUMENTS")

    if not os.path.exists(save_base + file_name):
        print("LOADING DOCUMENTS FROM URLS")
        loader = BeautifulSoupWebReader()
        documents = loader.load_data(urls=urls)  # may take some time
        with open(save_base + file_name, "wb") as file:
            pickle.dump(documents, file)
        print(f"Documents saved to {save_base + file_name}")
    else:
        print("LOADING DOCUMENTS FROM FILE")
        print(f"Opening {save_base + file_name}")
        with open(save_base + file_name, "rb") as file:
            documents = pickle.load(file)

    return objectify_documents_chroma(documents)


def chunk_documents(documents, chunk_size):
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs


def get_retrieval_df(current_retriever_vector_db_dir, llm, questions, documents, k_val, chain_type, search_type):
    embedding = OpenAIEmbeddings()

    retrieval_vectordb = Chroma.from_documents(
        documents=documents,
        persist_directory=current_retriever_vector_db_dir,
        embedding=embedding)

    retriever = retrieval_vectordb.as_retriever(search_type=search_type, search_kwargs={"k": k_val})

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type=chain_type,
        return_source_documents=True
    )

    queries, answers, context, scores, responses_latency = [], [], [], [], []

    for q in questions:
        try:
            time_start = time.time()
            docs = retrieval_vectordb.similarity_search_with_score(q, k=k_val)
            retry = 3  # Set the number of retry attempts
            q_scores, q_docs = [], []

            while retry > 0:
                try:
                    result = qa_chain({"query": q})
                    chain_docs = result['source_documents']
                    for cd in chain_docs:
                        q_docs.append(cd.page_content)
                    break  # Break the loop if the request is successful
                except openai.error.RateLimitError as rate_limit_error:
                    print(f"Rate limit reached. Waiting 10ms and retrying...")
                    time.sleep(0.01)  # Wait for 10ms
                    retry -= 1
                    if retry == 0:
                        raise rate_limit_error


            time_end = time.time()
            response_latency = time_end - time_start
            responses_latency.append(response_latency)

            for doc, score in docs:
                q_scores.append(score)
                #q_scores.append(1)

            answers.append(result['result'])
            queries.append(q)
            context.append(q_docs)
            scores.append(q_scores)



        except Exception as e:
            traceback.print_exc()
            print('Encountered an exception whilst running the chain: on query: ' + str(e))

    df = pd.DataFrame({
        'question': queries,
        'sampled_answer': answers,
        'response_latency': responses_latency,
        'context': context,
        'scores': scores
    })

    return df


def df_evals(
    df,
    eval_model,
    formatted_evals_column="retrieval_evals",
    qa_templ=templates.QA_PROMPT_TEMPLATE_STR,
):
    model = OpenAIModel(model_name=eval_model, temperature=0.0)
    Q_and_A_classifications = llm_eval_binary(
        dataframe=df,
        template=qa_templ,
        model=model,
        rails=["correct", "incorrect"],
    )
    df["qa_evals"] = Q_and_A_classifications
    df[formatted_evals_column] = run_relevance_eval(
        dataframe=df,
        query_column_name="question",
        retrieved_documents_column_name="context",
        template=RAG_RELEVANCY_PROMPT_TEMPLATE_STR,
        model=model,
        output_map={"relevant": 1, "irrelevant": 0},
        trace_data=False,
    )
    return df


# Performance metrics
def compute_precision_at_i(eval_scores, k):
    cpis = []
    print(k)
    print(eval_scores)
    for i in range(1, k + 1):
        cpis.append(sum(eval_scores[:i]) / i)
    return cpis


def compute_average_precision_at_i(evals, cpis, i):
    if np.sum(evals[:i]) == 0:
        return 0
    subset = cpis[:i]
    return (np.array(evals[:i]) @ np.array(subset)) / np.sum(evals[:i])


def get_rank(evals):
    for i, eval in enumerate(evals):
        if eval == 1:
            return i + 1
    return np.inf


def process_row(row, formatted_evals_column, k):
    print(row)
    formatted_evals = row[formatted_evals_column]
    cpis = compute_precision_at_i(formatted_evals, k)
    acpk = [compute_average_precision_at_i(formatted_evals, cpis, i) for i in range(1, k + 1)]
    ndcgis = [ndcg_score([formatted_evals], [row["scores"]], k=i) for i in range(1, k + 1)]
    ranki = [get_rank(formatted_evals[:i]) for i in range(1, k + 1)]
    return cpis + acpk + ndcgis + ranki


def calculate_metrics(df, k, formatted_evals_column="formatted_evals"):
    df["data"] = df.apply(lambda row: process_row(row, formatted_evals_column, k), axis=1)
    # Separate the list of data into separate columns
    derived_columns = (
        [f"context_precision_at_{i}" for i in range(1, k + 1)]
        + [f"average_context_precision_at_{i}" for i in range(1, k + 1)]
        + [f"ndcg_at_{i}" for i in range(1, k + 1)]
        + [f"rank_at_{i}" for i in range(1, k + 1)]
    )
    df_new = pd.DataFrame(df["data"].tolist(), columns=derived_columns, index=df.index)
    # Concatenate this new DataFrame with the old one:
    df_combined = pd.concat([df, df_new], axis=1)
    # don't want the 'data' column anymore:
    df_combined.drop("data", axis=1, inplace=True)
    return df_combined


def plot_graphs(all_data, k, save_dir="./", show=True, remove_zero=True):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plot_latency_graphs(all_data, save_dir, show)
    plot_mean_average_precision_graphs(all_data, save_dir, show, remove_zero)
    plot_mean_precision_graphs(all_data, save_dir, show, remove_zero)
    plot_ndcg_graphs(all_data, save_dir, show, remove_zero)
    plot_mrr_graphs(all_data, save_dir, show, remove_zero)
    plot_percentage_incorrect(all_data, save_dir, show, remove_zero)


def run_experiments(
        documents,
        questions,
        chunk_sizes,
        retrievers,
        k,
        save_dir,
        eval_model,
        all_data={}
):
    llm = ChatOpenAI(temperature=0)
    for chunk_size in chunk_sizes:
        if chunk_size not in all_data:
            all_data[chunk_size] = {}
        chunked_documents = chunk_documents(documents, chunk_size)
        for key, value in retrievers.items():
            if key not in all_data[chunk_size]:
                all_data[chunk_size][key] = {}
            for k_val in k:
                if k_val not in all_data[chunk_size][key]:
                    all_data[chunk_size][key][k_val] = {}
                current_retriever_vector_db_dir = save_dir + key + "/"
                df = get_retrieval_df(current_retriever_vector_db_dir, llm, questions, chunked_documents,
                                      k_val, value['chain_type'], value['search_type'])
                df = df_evals(df, eval_model)
                df = calculate_metrics(df, k_val, formatted_evals_column="retrieval_evals")
                all_data[chunk_size][key][k_val] = df
            tmp_save_dir = save_dir + "tmp_" + str(chunk_size) + "/"
            plot_graphs(all_data, k_val, tmp_save_dir, show=False)

            with open(tmp_save_dir + "data_all_data.pkl", "wb") as file:
                pickle.dump(all_data, file)
    return all_data


def main():
    openai.api_key = config.open_ai_key
    os.environ["OPENAI_API_KEY"] = config.open_ai_key
    web_title = "arize"  # nickname for this website, used for saving purposes
    base_url = "https://docs.arize.com/arize"
    # Local files
    file_name = "raw_documents.pkl"
    save_base = "./experiment_data/"
    now = datetime.datetime.now()
    run_name = now.strftime("%Y%m%d_%H%M")
    save_dir = save_base + run_name + "/"
    documents = load_documents(base_url, save_base, file_name)
    questions = read_strings_from_csv(
        "https://storage.googleapis.com/arize-assets/fixtures/Embeddings/GENERATIVE/constants.csv"
    )
    questions = questions[0:10]
    chunk_sizes = [900]
    k = [4]
    retrievers = config.retriever_types
    eval_model = "gpt-4"
    all_data = run_experiments(
        documents,
        questions,
        chunk_sizes,
        retrievers,
        k,
        save_dir,
        eval_model
    )


    with open(f"{save_dir}{web_title}_all_data.pkl", "wb") as f:
        pickle.dump(all_data, f)

    plot_graphs(all_data, k, save_dir + "/results_zero_removed/", show=False)
    plot_graphs(all_data, k, save_dir + "/results_no_zero_remove/", show=False, remove_zero=False)



if __name__ == "__main__":
    program_start = time.time()
    main()
    program_end = time.time()
    total_time = (program_end - program_start) / (60 * 60)
    print(f"EXPERIMENTS FINISHED: {total_time:.2f} hrs")
