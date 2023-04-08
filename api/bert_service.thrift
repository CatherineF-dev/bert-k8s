namespace java bert_service
namespace py bert_service

service BertEmbeddingService
{
	list<list<double>> get_embeddings(1: list<string> sentence),
}


