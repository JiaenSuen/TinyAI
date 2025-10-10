from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from vector import retriever


model = OllamaLLM(model="llama3")

template = """
You are an expert in answering questions about a pizza restaurant.
Use only the information from the provided customer reviews to answer the question.
If the reviews do not contain enough information, say "I'm not sure based on the reviews.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
Answer in a helpful and concise way.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


while True:
    print("\n\n")
    question = input("Ask your question (q to quit) : ")
    print("\n\n")
    if question == 'q': break

    reviews = retriever.invoke(question)
    result = chain.invoke({
        "reviews": reviews,
        "question": question,
    })
    print(result)