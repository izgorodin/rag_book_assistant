from setuptools import setup, find_packages

setup(
    name="rag_book_assistant",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'flask',
        'openai',
        'nltk',
        'numpy',
        'tqdm',
        'pinecone-client'
    ]
)

