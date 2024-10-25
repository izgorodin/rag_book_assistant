from typing import List, Dict, Any, NewType, Tuple
import numpy as np

# Базовые типы
Chunk = NewType('Chunk', str)
Score = NewType('Score', float)

# Эмбеддинги
Embedding = NewType('Embedding', np.ndarray)
EmbeddingList = NewType('EmbeddingList', List[Embedding])

# Результаты поиска
SearchResult = Dict[str, Any]
SearchResults = List[SearchResult]

# Типы для функций
QueryType = NewType('QueryType', str)
TopK = NewType('TopK', int)

# Типы для BM25
TokenizedChunk = List[str]
TokenizedChunks = List[TokenizedChunk]

# Типы для конфигурации
Config = Dict[str, Any]

# Типы для векторных операций
Vector = NewType('Vector', np.ndarray)
VectorList = List[Vector]

# Типы для индексов и идентификаторов
ChunkIndex = NewType('ChunkIndex', int)
ChunkId = NewType('ChunkId', str)

# Типы для результатов сравнения
SimilarityScore = NewType('SimilarityScore', float)
SimilarityScores = List[SimilarityScore]

# Типы для обработки естественного языка
Token = NewType('Token', str)
Tokens = List[Token]
POSTag = NewType('POSTag', str)
POSTags = List[POSTag]

# Типы для расширенных запросов
ExpandedQuery = NewType('ExpandedQuery', str)

# Типы для весов и параметров
Weight = NewType('Weight', float)