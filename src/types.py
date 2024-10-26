from typing import List, Dict, Any, NewType, Tuple, Callable, TypeVar, Generic, Union, Protocol, Literal
import numpy as np

# Базовые типы
Chunk = NewType('Chunk', str)
Score = NewType('Score', float)

# Эмбеддинги
Embedding = NewType('Embedding', np.ndarray)
EmbeddingList = NewType('EmbeddingList', List[Embedding])

# Результаты поиска
VectorDBResult = Dict[str, Any]
SearchResult = Dict[str, Any]
SearchResults = List[Union[SearchResult, VectorDBResult]]

# Типы для функций
T = TypeVar('T')
R = TypeVar('R')
EmbeddingFunction = Callable[[List[Chunk]], EmbeddingList]

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

# Add to types.py
PineconeIndex = NewType('PineconeIndex', Any)

# Добавляем протокол для стратегии поиска

# Добавляем типы для поиска
QueryType = NewType('QueryType', str)
TopK = NewType('TopK', int)

# Добавляем типы для BM25
BM25Scores = NewType('BM25Scores', np.ndarray)
EmbeddingScores = NewType('EmbeddingScores', np.ndarray)

# Добавляем типы для Pinecone
PineconeVector = Tuple[str, List[float], Dict[str, Any]]
PineconeVectors = List[PineconeVector]
PineconeQueryResult = Dict[str, Any]

# Добавляем типы для обработки текста
TokenWithPOS = Tuple[str, str]
TokensWithPOS = List[TokenWithPOS]

# Добавляем типы для результатов
ScoreArray = NewType('ScoreArray', np.ndarray)
CombinedScores = NewType('CombinedScores', np.ndarray)

# Добавляем типы для DataSource
DataSourceConfig = Dict[str, Any]
DataSourceResult = Dict[str, Any]

# OpenAI Service Types
ModelResponse = NewType('ModelResponse', str)
EmbeddingResponse = NewType('EmbeddingResponse', Dict[str, Any])
ChatMessage = Dict[str, str]
ChatMessages = List[ChatMessage]
ChatResponse = Dict[str, Any]

# OpenAI API Types
APIKey = NewType('APIKey', str)
ModelName = NewType('ModelName', str)
MaxTokens = NewType('MaxTokens', int)

# Service Response Types
ServiceResponse = Dict[str, Any]
ErrorResponse = Dict[str, str]
SuccessResponse = Dict[str, Any]

# OpenAI Specific Types
CompletionChoice = Dict[str, Any]
CompletionChoices = List[CompletionChoice]
EmbeddingData = Dict[str, Any]
EmbeddingDataList = List[EmbeddingData]
EmbeddingInput = NewType('EmbeddingInput', str)

# Types for RAG Context
Context = NewType('Context', str)
ContextMetadata = Dict[str, Any]
EnrichedContext = Dict[str, Union[Context, ContextMetadata]]

# Types for Search Strategy
SearchStrategyType = NewType('SearchStrategyType', str)
SearchStrategyName = Literal["simple", "hybrid", "semantic"]
SearchStrategyConfig = Dict[str, Any]

# Types for Answer Evaluation
GeneratedAnswer = NewType('GeneratedAnswer', str)
ReferenceAnswer = NewType('ReferenceAnswer', str)
QualityScore = NewType('QualityScore', float)

# Types for RAG Results
RelevantChunks = List[Dict[str, str]]
ChunkContent = NewType('ChunkContent', str)

# Types for Book Interface
BookContent = NewType('BookContent', str)
ProcessedBook = Dict[str, Union[List[Chunk], EmbeddingList]]
