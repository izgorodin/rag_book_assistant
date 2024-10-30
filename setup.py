from setuptools import setup, find_packages

setup(
    name="rag_book_assistant",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'openai',
        'python-dotenv',
        'numpy',
        'flask',
        'flask-socketio',
        'gunicorn',
        'eventlet',
        'pinecone-client',
        'colorlog',
        'python-docx',
        'pypdf',
        'odfpy',
        'nltk',
        'tqdm',
        'spacy',
        'rank_bm25',
        'scikit-learn'
    ],
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            'rag_book_assistant=src.cli:main',
        ],
    },
    author="Eduard Izgorodin",
    author_email="helloworld@uinside.org",
    description="A RAG-based book question answering system",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)