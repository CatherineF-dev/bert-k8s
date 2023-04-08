import glob
import sys
sys.path.append('./api/gen-py')

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

from bert_service import BertEmbeddingService

import seaborn as sns
from sklearn.metrics import pairwise

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

#@title Configure the model { run: "auto" }
BERT_MODEL = "https://tfhub.dev/google/experts/bert/wiki_books/2" # @param {type: "string"} ["https://tfhub.dev/google/experts/bert/wiki_books/2", "https://tfhub.dev/google/experts/bert/wiki_books/mnli/2", "https://tfhub.dev/google/experts/bert/wiki_books/qnli/2", "https://tfhub.dev/google/experts/bert/wiki_books/qqp/2", "https://tfhub.dev/google/experts/bert/wiki_books/squad2/2", "https://tfhub.dev/google/experts/bert/wiki_books/sst2/2",  "https://tfhub.dev/google/experts/bert/pubmed/2", "https://tfhub.dev/google/experts/bert/pubmed/squad2/2"]
# Preprocessing must match the model, but all the above use the same.
PREPROCESS_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

preprocess = hub.load(PREPROCESS_MODEL)
bert = hub.load(BERT_MODEL)

class BertEmbeddingHandler(BertEmbeddingService.Iface):
    def __init__(self):
        self.preprocess = hub.load(PREPROCESS_MODEL)
        self.bert = hub.load(BERT_MODEL)

    def get_embeddings(self, sentences):
        inputs = self.preprocess(sentences)
        outputs = self.bert(inputs)
        return outputs["pooled_output"]


if __name__ == '__main__':
    # Set up the thrift server
    handler = BertEmbeddingHandler()
    processor = BertEmbeddingService.Processor(handler)
    transport = TSocket.TServerSocket(host='127.0.0.1', port=9090)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    # Start the thrift server
    server = TServer.TThreadPoolServer(
        processor, transport, tfactory, pfactory)
    print('Starting the thrift server...')
    server.serve()
    print('done.')
