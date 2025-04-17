from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever
from typing import List
import Stemmer

def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    return [(s - min_score) / (max_score - min_score + 1e-9) for s in scores]

class WeightedFusionRetriever(BaseRetriever):
    def __init__(self, retriever_a, retriever_b, weight_a=0.6, top_k=5):
        self.retriever_a = retriever_a
        self.retriever_b = retriever_b
        self.weight_a = weight_a
        self.weight_b = 1.0 - weight_a
        self.top_k = top_k

    def _retrieve(self, query: str) -> List[NodeWithScore]:
        a_results = self.retriever_a.retrieve(query)
        b_results = self.retriever_b.retrieve(query)

        a_scores = normalize_scores([r.score for r in a_results])
        b_scores = normalize_scores([r.score for r in b_results])

        merged = {}
        for i, r in enumerate(a_results):
            merged[r.node.node_id] = (self.weight_a * a_scores[i], r.node)

        for i, r in enumerate(b_results):
            if r.node.node_id in merged:
                old_score, node = merged[r.node.node_id]
                merged[r.node.node_id] = (old_score + self.weight_b * b_scores[i], node)
            else:
                merged[r.node.node_id] = (self.weight_b * b_scores[i], r.node)

        result = [NodeWithScore(node=n, score=s) for _, (s, n) in merged.items()]
        result.sort(key=lambda x: x.score, reverse=True)
        return result[:self.top_k]

def get_hybrid_retriever(index, nodes):
    vector_retriever = index.as_retriever(similarity_top_k=5)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        stemmer=Stemmer.Stemmer("english"),
        similarity_top_k=5,
        language="english"
    )
    return WeightedFusionRetriever(vector_retriever, bm25_retriever, weight_a=0.6)
