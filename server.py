import json
import torch
from transformers import PreTrainedModel, BertConfig, BertModel, AutoTokenizer, RobertaConfig, RobertaModel, \
    RobertaTokenizer
import numpy as np
from pyserini.search import SimpleSearcher
from pyserini.analysis import Analyzer, get_lucene_analyzer
from pyserini.index import IndexReader
from tqdm import tqdm, tqdm_notebook
import math
import sys
from socket import *
from time import *
from _thread import *
import torch
from ANCE import AnceModel

# Load ANCE model from local files
print("Loading ANCE model")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ance_model = AnceModel.from_pretrained('ANCE_Model').eval()
ance_model.to(device)
ance_tokenizer = AutoTokenizer.from_pretrained('ANCE_Model')

# Setup server socket
server_socket = socket(AF_INET, SOCK_STREAM)
host = '127.0.0.1'

# Load indexed documents
stemming = None
stopwords = False
index = 'indexes/lucene-index-msmarco-passage-vectors-noProcessing/'

lucene_analyzer = get_lucene_analyzer(stemming=stemming, stopwords=stopwords)
analyzer = Analyzer(lucene_analyzer)
searcher = SimpleSearcher(index)
searcher.set_analyzer(lucene_analyzer)
index_reader = IndexReader(index)

print("Create document frequency dictionary to speed up scoring later, this will take around 2 min.")
df_dict = {}
for term in tqdm(index_reader.terms(), desc="loading idf dictionary:"):
    df_dict[term.term] = term.df

print("Cache document length and docids for the collection, this will take around 2 min.")
doc_len_dict = {}
doc_id_dict = {}
with open('collection.tsv', 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines, desc="loading doc_length dictionary:"):
        docid, text = line.split('\t')
        doc_len_dict[docid] = len(text.split())
        internal_id = index_reader.convert_collection_docid_to_internal_docid(docid)
        doc_id_dict[internal_id] = docid

# Main function for searching
def search(query: str, k: int = 50, scorer=None):
    assert scorer is not None
    # get the analyzed term list
    q_terms = analyzer.analyze(query)
    doc_socres = {}
    for term in q_terms:
        # get the posting list for the current term
        postings_list = index_reader.get_postings_list(term, analyzer=None)
        if postings_list is not None:
            # get the document frequency of the current term
            df = df_dict[term]
            # iterate the posting list
            for posting in tqdm(postings_list, desc=f"Iterate posting for term '{term}'"):
                internal_id = posting.docid
                docid = doc_id_dict[internal_id]
                tf = posting.tf
                doc_len = doc_len_dict[docid]
                score = scorer(tf, df, doc_len)
                if docid in doc_socres.keys():
                    doc_socres[docid] += score
                else:
                    doc_socres[docid] = score

    # Sort the results by the score.
    results = [(docid, doc_socre) for docid, doc_socre in doc_socres.items()]
    results = sorted(results, key=lambda x: x[1], reverse=True)[:k]
    return results

# Parameters for BM25
N = index_reader.stats()["documents"]
avg_dl = index_reader.stats()["total_terms"] / index_reader.stats()["documents"]
k_1 = 1.2
b = 0.75


# My implement of BM25 scoring
def bm25(tf, df, doc_len):
    idf = math.log10(N / (df + 1))
    a1 = idf * tf * (k_1 + 1)
    a2 = tf + k_1 * (1 - b + b * doc_len / avg_dl)
    # return round(a1 / a2, 3)
    return a1 / a2

# My implement of these algorithms
def prf_query_expansion(query: str, n: int, m: int):
    hits = searcher.search(query, k=50)
    expanded_query = query + ' '
    bm25_score = {}
    for i in range(0, n):
        word_freq = index_reader.get_document_vector(hits[i].docid)
        bm25_set = {term: index_reader.compute_bm25_term_weight(hits[i].docid,
                                                                term,
                                                                analyzer=None)
                    for term in word_freq.keys()}
        for word in word_freq:
            if word not in bm25_score:
                bm25_score[word] = 0
            bm25_score[word] += bm25_set[word]

    a1 = sorted(bm25_score.items(), key=lambda x: x[1], reverse=True)
    print(a1[:10][1])
    for word in a1[:m]:
        expanded_query += word[0] + ' '
    expanded_query = expanded_query[:-1]
    return expanded_query


def idfr_query_reduction(query: str, n: int):
    terms = analyzer.analyze(query)
    pruned_query = ''
    df = {}
    for term in terms:
        quan = (index_reader.get_term_counts(term, analyzer=None))[0]
        df[term] = quan
    words = sorted(df.items(), key=lambda x: x[1])
    for word in words[:n]:
        # Re-construct the query
        pruned_query += word[0] + ' '
    pruned_query = pruned_query[:-1]
    return pruned_query


def borda(runs):
    seen = {}
    for topic, results in runs.items():
        if topic not in seen:
            seen[topic] = {}
        for i, docid in enumerate(results.keys()):
            n = len(results)
            rd = i
            score = (n - rd + 1) / n
            if docid not in seen[topic]:
                seen[topic][docid] = score
            else:
                seen[topic][docid] += score
    return seen


def combsum(runs):
    seen = {}
    for topic, results in runs.items():
        if topic not in seen:
            seen[topic] = {}
        for docid, score in results.items():
            if docid not in seen[topic]:
                seen[topic][docid] = score
            else:
                seen[topic][docid] += score
    return seen


def combmnz(runs):
    seen = {}
    count = {}
    for topic, results in runs.items():
        if topic not in seen:
            seen[topic] = {}
            count[topic] = {}

        for docid, score in results.items():
            if score > 0:
                if docid not in count[topic]:
                    count[topic][docid] = 0
                else:
                    count[topic][docid] += 1

            if docid not in seen[topic]:
                seen[topic][docid] = score
            else:
                seen[topic][docid] += score

        for docid, score in results.items():
            if docid in count[topic]:
                seen[topic][docid] += score * count[topic][docid]
            else:
                seen[topic][docid] += 0

    return seen

# Re-rank the results
def ance_rerank(query, pre_result, k):
    run = []
    # Tokenlize the query
    inputs = ance_tokenizer([query],
                            max_length=64,
                            padding='longest',
                            truncation=True,
                            add_special_tokens=True,
                            return_tensors='pt').to(device)
    query_embeddings = ance_model(inputs["input_ids"]).detach().cpu().numpy().flatten()
    doc_list = [x[0] for x in pre_result]

    for doc_id in doc_list[:k]:
        res = json.loads(searcher.doc(doc_id).raw())
        # Compute new scores
        passage = res['contents']
        passage_inputs = ance_tokenizer(
            [passage],
            max_length=512,
            padding='longest',
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        passage_inputs.to(device)
        passage_embeddings = ance_model(passage_inputs["input_ids"]).detach().cpu().numpy().flatten()
        score = np.dot(query_embeddings, passage_embeddings)
        run.append((doc_id, score))

    sorted_run = sorted(run, key=lambda x: x[1], reverse=True)
    return sorted_run


def send_message(connection, message):
    message = message.encode('utf-8')
    connection.sendall(message)


def threaded_client(connection, cfg):
    running = True
    # Load parameters from config file
    num_res = cfg['results']
    method = cfg['method']
    expansion = cfg["expansion"]
    max_len = cfg["max_length"]

    while running:
        origin_query = connection.recv(1024 * 16).decode('utf-8')
        len_query = len(origin_query.split(' '))
        if len_query > max_len:
            query = idfr_query_reduction(origin_query, 20)
            message = "The length limitation for query is {} words, hence the query was processed.".format(str(max_len))
            send_message(connection, message)
        elif len_query < 3:
            if expansion:
                query = prf_query_expansion(origin_query, 3, 3)
            else:
                query = origin_query
        else:
            query = origin_query
        
        if query:
            results = search(query, num_res, scorer=bm25)
                
            # Enable re-rank
            if method == 'ance':
                results = ance_rerank(query, results, num_res)

            print("Query:", query)
            print("Result:", results)
            result_text = ''
            i = 1
            for pair in results:
                res = json.loads(searcher.doc(pair[0]).raw())
                passage = res['contents']
                result_text += 'Result: {} '.format(str(i)) + passage + '\n'
                i += 1
            send_message(connection, result_text)


def startServer(port, config):
    with open(config, 'r') as fs:
        cfg = json.load(fp=fs)

    server_socket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen()
    print("Server running on", str(host) + ":" + str(port))

    while True:
        client_socket, client_addr = server_socket.accept()
        print("Connection from:", client_addr)
        start_new_thread(threaded_client, (client_socket, cfg))


if __name__ == '__main__':
    startServer(int(sys.argv[1]), sys.argv[2])
