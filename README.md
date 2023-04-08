# bert-k8s

Deploy bert nlp model as a service in a k8s cluster.

## Deploy bert embedding inference in a k8s cluster

1. Build image and push into registry
```bash
REGISTRY=gcr.io/your_registry/bert:dev

git clone https://github.com/CatherineF-dev/bert-k8s.git
cd bert-k8s
docker build . --tag ${REGISTRY}
docker push ${REGISTRY}
```

2. Deploy in k8s cluster
```bash
# replace image
sed -r -i "s|gcr.io/your_registry/bert:dev|$REGISTRY|g" yamls/deployment.yaml
kubectl apply -f yamls
```

3. Get embedding results
```python
#!/usr/bin/env python
# client.py

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
print(embeddings[0][:20], embeddings[1][:20]) # [-0.47659167647361755, 0.41174161434173584, 0.5765763521194458, 0.8259852528572083, 0.44902855157852173, 0.6287587285041809, 0.436123788356781, -0.9251144528388977, 0.8255095481872559, 0.6757315993309021, 0.11731564998626709, -0.8891088962554932, 0.2539674639701843, -0.4233919084072113, -0.1866266131401062, 0.952871561050415, 0.7946820855140686, 0.6445589661598206, -0.8728901743888855, 0.7713952660560608] [-0.6382293105125427, 0.53094083070755, 0.5713736414909363, 0.8513802886009216, 0.5705713629722595, 0.5697867274284363, 0.333121657371521, -0.9574463963508606, 0.8735503554344177, 0.646080493927002, 0.3136337101459503, -0.915554404258728, 0.2994466722011566, -0.32557645440101624, 0.04537038877606392, 0.9275534152984619, 0.666790783405304, 0.6805806756019592, -0.9311433434486389, 0.7361170649528503]

# Close the thrift client
transport.close()
```

```bash
# port-forward
$ kubectl port-forward $(kubectl get pod --selector="app=bert" --output jsonpath='{.items[0].metadata.name}') 9090:9090

$ python3 client.py
[-0.4765907824039459, 0.41174113750457764, 0.5765762329101562, 0.8259849548339844, 0.44902709126472473, 0.6287593841552734, 0.43612444400787354, -0.9251151084899902, 0.8255098462104797, 0.6757326126098633, 0.11731579154729843, -0.8891090154647827, 0.2539665102958679, -0.42338892817497253, -0.18662399053573608, 0.9528712630271912, 0.7946816086769104, 0.6445597410202026, -0.8728898167610168, 0.7713953256607056] [-0.6382287740707397, 0.5309401750564575, 0.5713725686073303, 0.8513800501823425, 0.5705699324607849, 0.5697899460792542, 0.33312463760375977, -0.9574467539787292, 0.873551070690155, 0.6460804343223572, 0.3136363923549652, -0.9155545830726624, 0.29944944381713867, -0.3255760669708252, 0.04537119343876839, 0.9275529384613037, 0.6667917370796204, 0.68058180809021, -0.9311428666114807, 0.7361184358596802]
```

