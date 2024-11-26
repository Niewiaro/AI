# Chat-Based Question Answering System with LangChain and Transformers

This project is a simple implementation of a **Question Answering System** built using LangChain and Transformers. It utilizes document embedding, vector search, and a question-answering pipeline to extract answers from a given text document.

## Features

- **Document Parsing:** Loads a text document and splits it into manageable chunks.
- **Vector Search:** Uses FAISS for efficient similarity search within the document.
- **Embeddings:** Employs HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` for text embeddings.
- **Question Answering:** Utilizes `distilbert-base-cased-distilled-squad` for answering questions based on the retrieved context.
- **Extensible Configuration:** Customize questions and documents easily through the `Config` class.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Niewiaro/AI-RAG.git
   cd AI-RAG
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Configuration

Edit the `Config` class in the script to set your custom document and questions:
```python
class Config:
    def __init__(self):
        self.questions = (
            "Who am I writing with?",
            "When will he arrive?",
            "Tell me his favorite song",
        )
        self.file_name = "document.txt"
        self.document = """
        Your document content here...
        """
```

### Running the Script

1. Run the script:
   ```bash
   python main.py
   ```

2. The script will:
   - Save the document as a text file.
   - Build a vector store from the document.
   - Execute the specified questions.
   - Print the answers to the console.

---

## Example Output

For the given example document and questions, the output might look like this:

```
Q: Who am I writing with?
A: Michał

Q: When will he arrive?
A: 5:55 PM

Q: Tell me his favorite song
A: Mateusz, flip the switch
```

---

## Project Structure

- `main.py`: The main script containing the logic for loading documents, building a vector store, and executing the question-answering pipeline.
- `document.txt`: The sample document file created by the script.

---

## Technologies Used

- **[LangChain](https://github.com/hwchase17/langchain):** Framework for working with language models and text embeddings.
- **[Transformers](https://github.com/huggingface/transformers):** HuggingFace's library for NLP models.
- **[FAISS](https://github.com/facebookresearch/faiss):** Library for efficient similarity search.

---

## Future Enhancements

- Add support for more languages using multilingual models.
- Improve accuracy by fine-tuning the QA model on specific domains.
- Integrate with a front-end interface for user-friendly interaction.

---

## Acknowledgements

Special thanks to:
- [HuggingFace](https://huggingface.co/) for providing state-of-the-art models.
- [LangChain](https://langchain.com/) for making it easier to work with LLMs and embeddings. 
