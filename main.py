import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

file_name = "document.txt"
document = """
**What time do you have Barbara?**  
*Michał:*  
Barbera?  
*Sent:*  
Already done.  

---

**Thu, 17:58**  
*Sent:*  
Hey.  
*Sent:*  
There’s a buyer for the car.  
*Sent:*  
I don’t know if I’ll make it in time.  

---

**Thu, 18:19**  
*Michał:*  
Alright.  
*Michał:*  
Cool.  
*Michał:*  
Good luck.  
*Michał:*  
I’m going as usual.  
*Michał:*  
And then I’m heading straight to Sara’s.  
*Michał:*  
Mud.  
*Michał:*  
Bolt.  

---

**Sun, 18:02**  
*Michał:*  
F**k, xD. I park, there’s a great spot, and I see a guy who’s probably leaving because the lights are on. I’m standing there for 20 seconds, waiting, flashing my high beams so he knows.  
*Michał:*  
Another 20 seconds go by, and I realize something’s wrong because there’s no smoke coming from the back.  
*Michał:*  
Turns out the guy just left the lights on, xD.  
*Michał:*  
And there’s nobody inside, lol.  

---

**Sun, 18:25**  
*Sent:*  
XDDD  

---

**Sun, 23:02**  
*Michał:*  
I’m listening 24/7 "Mateusz, flip the switch" so Spotify counts this as my most-played song in 2024.  
*Michał:*  
The worst part is that I’m starting to like it more and more.  
*Michał:*  
And the worst part is that I just realized Spotify counts it only until the end of October.  
*Michał:*  
What a bunch of f**kers, lol.  
*Michał:*  
I’m dropping a deuce and then heading out.  
*Michał:*  
So, I’ll be there around 5:55 PM.  
*Michał:*  
Mhmmm, I thought it wasn’t our building anymore.  

---

**19:27**  
*Michał:*  
Send me a fart from your p***y.  
"""


def main() -> None:
    with open(file_name, "w") as file:
        file.write(document)

    loader = TextLoader(file_name)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=30)
    texts = text_splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(texts, embedding)

    model = pipeline(
        "question-answering", model="distilbert/distilbert-base-cased-distilled-squad"
    )

    def run_rag_system(query):
        docs = vector_store.similarity_search(query=query, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        result = model(question=query, context=context)
        answer = result["answer"]
        return answer

    question = "When Michał arrive?"
    print(f"Q:\t{question}\nA:\t{run_rag_system(question)}")


if __name__ == "__main__":
    main()
