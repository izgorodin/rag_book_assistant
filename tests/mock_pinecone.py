from typing import List, Dict, Any

class MockPineconeIndex:
    def __init__(self):
        self.vectors = {}

    def upsert(self, vectors: List[tuple]):
        for id, embedding, metadata in vectors:
            self.vectors[id] = (embedding, metadata)

    def query(self, vector: List[float], top_k: int = 1, include_metadata: bool = True) -> Dict[str, Any]:
        matches = [{"id": id, "score": 0.9, "metadata": metadata} 
                   for id, (_, metadata) in list(self.vectors.items())[:top_k]]
        return {"matches": matches}

    def delete(self, delete_all: bool = False):
        if delete_all:
            self.vectors.clear()

class MockPinecone:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.indexes = {}

    def list_indexes(self):
        class MockIndexList:
            def __init__(self, names):
                self._names = names
            def names(self):
                return self._names
        return MockIndexList(list(self.indexes.keys()))

    def create_index(self, name: str, dimension: int, metric: str, spec: Any):
        self.indexes[name] = MockPineconeIndex()

    def Index(self, name: str) -> MockPineconeIndex:
        if name not in self.indexes:
            self.indexes[name] = MockPineconeIndex()
        return self.indexes[name]

    def reset(self):
        self.indexes.clear()
