import asyncio
from fastapi import FastAPI, HTTPException

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
        response_messages = await agent.run(request.message)
        return ChatResponse(messages=response_messages)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "OK"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    