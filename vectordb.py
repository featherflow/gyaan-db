from typing import List, Tuple

import numpy as np
from embedding import BaseEmbedding


class IndexDocument:
    """
    Indexes the chunks, creates dictionary with text, vectors and other metadata.
    Returns the List of dictionary and vector embedding matrix
    """

    def __init__(self, chunks: List[str], embedding_object: BaseEmbedding):
        self.chunks = chunks
        self.embedding_object = embedding_object

    def indexed_document(self) -> Tuple[List[str], np.ndarray]:
        indexes, index_matrix = [], []

        for chunk in self.chunks:
            chunk_vector = self.embedding_object.get_embedding(chunk)
            indexes.append(chunk)
            index_matrix.append(chunk_vector)
        index_matrix = np.array(index_matrix)

        return indexes, index_matrix


class TopChunks:
    """
    Based on the query chunk and the indexed document. Return the chunks with the best similarity to the question
    """

    def __init__(
        self,
        indexes: List[dict],
        index_matrix: np.ndarray,
        embedding_obj: BaseEmbedding,
        metric: str = "cosine",
    ):
        self.indexes = indexes
        self.indexes_matrix = index_matrix
        self.embedding_obj = embedding_obj
        self.metric = metric

        if self.metric != "cosine":
            raise Exception("Metric is not supported: ", self.metric)

    def cosine_similarity(self, query_vector: np.ndarray):
        "Cosine similarity"
        cosines = np.matmul(self.indexes_matrix, query_vector) / (
            np.linalg.norm(self.indexes_matrix, axis=1) * np.linalg.norm(query_vector)
        )
        return cosines

    def top_k(self, query: str, k: int = 5) -> List[str]:
        "Top k chunks based on similarity metric"
        query_vector = np.asarray(self.embedding_obj.get_embedding(query))
        cosine_vector = self.cosine_similarity(query_vector)
        top_indices = np.argsort(cosine_vector)[::-1][:k]
        top_chunks = [self.indexes[i] for i in list(top_indices)]
        return top_chunks

    def top_threshold(self, query: str, threshold: float = 0.9) -> List[str]:
        "Top chunks passing the similarity threshold"
        raise Exception("Top Threshold Not Implemented")

    def top_k_threshold(
        self, query: str, k: int = 5, threshold: float = 0.9
    ) -> List[str]:
        "Top k chunks passing the threshold"
        raise Exception("Top K Threshold Not Implemented")

