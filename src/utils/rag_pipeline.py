import ollama

from src.utils.retriever import retrieve_knowledge


def answer_agriculture_question(question, top_k=3):
    """
    Answer an agricultural question using
    FarmSense AI's local RAG pipeline.

    Pipeline:
    User Question
        ↓
    ChromaDB Retrieval
        ↓
    Retrieved Agricultural Knowledge
        ↓
    Local LLM (Ollama)
        ↓
    Final Answer
    """

    # Retrieve relevant knowledge
    retrieved_results = retrieve_knowledge(
        question,
        top_k=top_k
    )

    # Combine retrieved documents
    context_parts = []

    for result in retrieved_results:
        context_parts.append(
            result["document"]
        )

    context = "\n\n".join(context_parts)

    # System instructions for the local LLM
    system_prompt = """
You are FarmSense AI, an agricultural
information assistant.

Answer the user's question using the
provided agricultural knowledge.

Important rules:

1. Prefer the provided knowledge over
   unsupported assumptions.

2. Do not invent agricultural facts.

3. If the provided knowledge does not
   contain enough information, clearly say
   that the available knowledge base does
   not contain enough information.

4. Give practical, easy-to-understand
   explanations.

5. For disease-management questions,
   remind users that image-based AI
   predictions are not definitive diagnoses.

6. Do not provide unsafe pesticide or
   chemical instructions.

7. Recommend following local agricultural
   extension guidance and product labels
   when agricultural products are involved.
"""

    # User prompt containing retrieved knowledge
    user_prompt = f"""
Agricultural knowledge retrieved from
the FarmSense AI knowledge base:

--------------------
{context}
--------------------

User question:

{question}

Answer the question using the retrieved
knowledge above.
"""

    # Run local LLM through Ollama
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    # Extract local model response
    answer = response["message"]["content"]

    return {
        "answer": answer,
        "sources": retrieved_results
    }


# Command-line testing
if __name__ == "__main__":

    question = input(
        "\nAsk FarmSense AI: "
    ).strip()

    result = answer_agriculture_question(
        question
    )

    print(
        "\n================================"
    )
    print(
        "FARMSENSE AI ANSWER"
    )
    print(
        "================================"
    )

    print(
        "\n" + result["answer"]
    )

    print(
        "\n================================"
    )
    print(
        "RETRIEVED SOURCES"
    )
    print(
        "================================"
    )

    for index, source in enumerate(
        result["sources"],
        start=1
    ):
        print(
            f"\n--- Source {index} ---"
        )

        print(
            source["document"]
        )

        print(
            f"\nDistance: "
            f"{source['distance']:.4f}"
        )
