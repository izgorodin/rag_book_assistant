from setuptools import setup, find_packages

setup(
    name="rag-book-qa",
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
            'rag-book-qa=src.cli:main',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
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