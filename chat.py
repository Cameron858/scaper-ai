import warnings

import gradio as gr
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pyprojroot import here
from starlette.exceptions import StarletteDeprecationWarning

from scaper_ai import settings

SYS_PROMPT = """
You are a helpful assistant for fishkeeping and aquarium care.

You will be given a <context> block containing reference material.

Rules:
- Only use the context if it is relevant.
- If no context is provided, ignore it completely.
- Never mention:
  - documents
  - sources
  - retrieval
  - context blocks
- Do not add notes, disclaimers, or meta-commentary.
- Answer directly and concisely.
- If you do not know the answer, say "I don't know".
"""

embeddings = OllamaEmbeddings(model=settings.embed_model)

vector_store = Chroma(
    collection_name="caresheets",
    embedding_function=embeddings,
    persist_directory=str(here("db/chroma_db")),
)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.4,
    },
)

chat = ChatOllama(model=settings.chat_model)


def response(message, history):

    global chat, retriever

    docs = retriever.invoke(message)

    if docs:
        context = "\n\n---\n\n".join(
            f"SOURCE: {doc.metadata.get('source', 'Unknown')}\n"
            f"{doc.page_content.strip()}"
            for doc in docs
        )

        sources = "\n".join(
            f"ID: {doc.id} - SOURCE: {doc.metadata.get('source', 'Unknown')}"
            for doc in docs
        )

        user_msg = f"""
            <context>
            {context}
            </context>

            {message}
        """.strip()

    else:
        context = None
        sources = "N/A"
        user_msg = message

    # convert between gradio chat and ollama
    # {'role': 'user', 'metadata': None, 'content': [{'text': 'Yo', 'type': 'text'}], 'options': None} -> {"role": "user", "content": "Yo"}
    formatted_history = []
    for record in history:
        formatted_history.append(
            {"role": record["role"], "content": record["content"][0]["text"]}
        )

    messages = [
        {"role": "system", "content": SYS_PROMPT},
        *formatted_history,
        {"role": "user", "content": user_msg},
    ]

    result = chat.invoke(messages)

    debug_content = f"\n---\n```text\nSources:\n{sources}\n```"

    return f"{result.content}{debug_content}"


def main():
    gr.ChatInterface(
        fn=response,
        title="Scaper AI",
        description="A chatbot for fishkeeping and aquarium care",
    ).launch()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=StarletteDeprecationWarning)
        main()
