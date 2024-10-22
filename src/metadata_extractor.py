import spacy
from typing import Dict, List, Any
from textblob import TextBlob

nlp = spacy.load("en_core_web_sm")

class TextMetadata:
    def __init__(self, text: str):
        self.text = text
        self.entities: Dict[str, List[str]] = {}
        self.key_phrases: List[str] = []
        self.dates: List[str] = []
        self.topics: List[str] = []
        self.sentiment: float = 0.0
        self.importance_score: float = 0.0
        self.source: str = ""

    def extract_metadata(self):
        doc = nlp(self.text)
        
        # Извлечение сущностей
        for ent in doc.ents:
            if ent.label_ not in self.entities:
                self.entities[ent.label_] = []
            self.entities[ent.label_].append(ent.text)
        
        # Извлечение ключевых фраз (используем именные фразы как прокси)
        self.key_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Извлечение дат
        self.dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        
        # Извлечение тем (используем корневые слова предложений как прокси)
        self.topics = list(set([sent.root.text for sent in doc.sents]))
        
        # Анализ настроения
        blob = TextBlob(self.text)
        self.sentiment = blob.sentiment.polarity
        
        # Простой подсчет важности (можно улучшить)
        self.importance_score = len(self.entities) + len(self.key_phrases) + len(self.dates)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": self.entities,
            "key_phrases": self.key_phrases,
            "dates": self.dates,
            "topics": self.topics,
            "sentiment": self.sentiment,
            "importance_score": self.importance_score,
            "source": self.source
        }

def extract_metadata(text: str) -> Dict[str, Any]:
    metadata = TextMetadata(text)
    metadata.extract_metadata()
    return metadata.to_dict()