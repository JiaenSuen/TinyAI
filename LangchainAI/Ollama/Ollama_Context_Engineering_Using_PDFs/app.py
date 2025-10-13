# langchain llama3 : context engineering by using pdf files
from langchain_ollama import OllamaLLM
from langchain.chains.retrieval_qa.base import RetrievalQA
from vector_database import get_retriever, init_vector_database

def query(question: str):
    init_vector_database()
    retriever = get_retriever()
    llm = OllamaLLM(model="llama3")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        verbose=False
    )

    answer = qa.invoke({"query": question})
    return answer["result"]


# Quest & Expect Answer  
Question_List = [
    "What are the main foods that red foxes eat according to the fact sheet?",
    "What are the different color variations of red foxes mentioned in the document?",
    "How far can male red foxes travel during dispersal?",
    "What types of sounds do red foxes make to communicate?",
    "What is the typical litter size of red foxes?",
    "How do red foxes store food for later use?",
    "What is the conservation status of red foxes in Canada?",
    "What role do red foxes play in controlling rabies?",
    "What is a fun fact about red fox behavior mentioned in the fact sheet?",
    "Does the fact sheet mention the exact weight of an adult red fox?"
]

if __name__ == "__main__":
    output_file = "output.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for i, question in enumerate(Question_List, 1):
            print(f"Processing Question {i}: {question}")
            answer = query(question)
            f.write(f"Question {i}: {question}\n")
            f.write(f"Answer: {answer}\n")
            f.write("-" * 50 + "\n")
            print(f"AI Bot: {answer}\n")
    print(f"All answers have been written to {output_file}")