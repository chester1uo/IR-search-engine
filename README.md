Search engine

Dataset: zyznull/msmarco-passage-corpus, Download: https://huggingface.co/datasets/zyznull/msmarco-passage-corpus



Libraries:

Pyserini https://github.com/castorini/pyserini

ANCE https://github.com/microsoft/ANCE



Before running:

Install the open-jdk: https://openjdk.org/install/

Install the necessary packages

```
pip install -r requirements.txt
```



Usage:

(1) Preparing the dataset

Covert to JSON:

```bash
python3 tools/convert_collection_to_jsonl.py \
 --collection-path collection.tsv \
 --output-folder collection/collection_jsonl
```

Index the document:

```bash
python3 -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
 -threads 12 -input collection/collection_jsonl \
 -index indexes/lucene-index-msmarco-passage-vectors-noProcessing \
 -optimize \
 -storeRaw \
 -stemmer none \
 -keepStopwords \
 -storePositions \
 -storeDocvectors 
```

(2) Start the search engine server

```bash
python3 server.py PORT config_path
```

To enable it runs background:

```
nohup python3 server.py PORT config_path &
```



(3) Start the client

The server address is 127.0.0.1 is you test on local machine. The client port can be any valid port.

```
python3 client.py server_address server_port client_port
```

