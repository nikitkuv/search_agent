import asyncio
from fastapi import FastAPI, HTTPException
from uuid import uuid4
from typing import List

from settings import Settings
from schemas import ChatRequest, ChatResponse
from search_agent import State, SearchAgent


app = FastAPI(
    title="AI Search Agent API",
    description="An agent that uses LLM + Tavily Search to answer questions.",
    version="1.0.0"
)


agent = SearchAgent()



@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        thread_id = request.thread_id or str(uuid4())
        response_messages = await agent.run(request.message, thread_id)
        return ChatResponse(messages=response_messages, thread_id=thread_id)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/conversation/{thread_id}", response_model=ChatResponse)
def get_conversation(thread_id: str):
    try:
        messages = agent.get_conversation(thread_id)
        return ChatResponse(messages=messages, thread_id=thread_id)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/conversations", response_model=List[str])
def list_conversations():
    try:
        thread_ids = agent.list_conversations()
        return thread_ids
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list conversations: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "OK"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    