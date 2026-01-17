import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None):
        self.vectors = defaultdict(np.array)
        self.embedding_model = embedding_model or EmbeddingModel()
        self.metadata = {}

    def insert(self, key: str, vector: np.array, metadata: dict = None) -> None:
        self.vectors[key] = vector
        if metadata is not None:
            self.metadata[key] = metadata

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = cosine_similarity,
        metadata_filter: dict = None
    ) -> List[Tuple[str, float]]:
        # Filter vectors by metadata if filter is provided
        items_to_search = self.vectors.items()
        if metadata_filter:
            items_to_search = [
                (key, vector) for key, vector in self.vectors.items()
                if self._matches_metadata_filter(key, metadata_filter)
            ]
        
        scores = [
            (key, distance_measure(query_vector, vector))
            for key, vector in items_to_search
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def _matches_metadata_filter(self, key: str, metadata_filter: dict) -> bool:
        """Check if metadata for a key matches the filter."""
        if key not in self.metadata:
            return False
        for filter_key, filter_value in metadata_filter.items():
            if self.metadata[key].get(filter_key) != filter_value:
                return False
        return True

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_as_text: bool = False,
        metadata_filter: dict = None,
    ) -> List[Tuple[str, float]]:
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure, metadata_filter)
        return [result[0] for result in results] if return_as_text else results

    def retrieve_from_key(self, key: str) -> np.array:
        return self.vectors.get(key, None)

    async def abuild_from_list(self, list_of_text: List[str], metadata_list: List[dict] = None) -> "VectorDatabase":
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for i, (text, embedding) in enumerate(zip(list_of_text, embeddings)):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
            self.insert(text, np.array(embedding), metadata)
        return self


if __name__ == "__main__":
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]

    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text))
    k = 2

    searched_vector = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"Closest {k} vector(s):", searched_vector)

    retrieved_vector = vector_db.retrieve_from_key(
        "I like to eat broccoli and bananas."
    )
    print("Retrieved vector:", retrieved_vector)

    relevant_texts = vector_db.search_by_text(
        "I think fruit is awesome!", k=k, return_as_text=True
    )
    print(f"Closest {k} text(s):", relevant_texts)
