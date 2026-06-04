from fastapi import FastAPI
from pydantic import BaseModel

from rag_chat import (
    retrieve_context,
    build_prompt,
    ask_llm,
)

app = FastAPI()


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(request: ChatRequest):
    plants, context = retrieve_context(request.question)

    prompt = build_prompt(request.question, plants, context)

    answer = ask_llm(prompt)

    return {"answer": answer}
