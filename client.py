#!/usr/bin/env python

import sys
sys.path.append('./api/gen-py')

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from bert_service import BertEmbeddingService


# Set up the thrift client
transport = TSocket.TSocket('localhost', 9090)
transport = TTransport.TBufferedTransport(transport)
protocol = TBinaryProtocol.TBinaryProtocol(transport)
client = BertEmbeddingService.Client(protocol)
transport.open()

# Call the get_embeddings function
sentences = ['This is a sentence.', 'This is another sentence.']
embeddings = client.get_embeddings(sentences)
print(embeddings[0][:20], embeddings[1][:20])

# Close the thrift client
transport.close()
