import asyncio

from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict

from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.postgres import PostgresStore
import psycopg

from settings import Settings


class State(TypedDict):
    messages: Annotated[list, add_messages]


class SearchAgent:
    def __init__(
        self, 
        model_name: str = Settings.model_name, 
        temperature: float = Settings.temperature,
        max_results: int = Settings.max_results,
        database_url: str = Settings.db_url
    ):
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url="http://host.docker.internal:11434"
        )
        self.tool = TavilySearch(max_results=max_results, api_key=Settings.TAVILY_API_KEY)
        self.llm_with_tools = self.llm.bind_tools([self.tool])

        self.database_url = database_url
        conn = psycopg.connect(self.database_url)
        conn.autocommit = True
        self.store = PostgresStore(conn)
        self.store.setup()

        self.graph = self._create_compile_graph()


    async def _llm_call_node(self, state: State) -> Dict[str, Any]:
        results = await self.llm_with_tools.ainvoke(state["messages"])
        return {"messages": [results]}

    
    def _create_compile_graph(self) -> StateGraph:
        graph = StateGraph(State)

        graph.add_node("llm", self._llm_call_node)
        graph.add_node("tools", ToolNode(tools=[self.tool]))

        graph.add_edge(START, "llm")
        graph.add_conditional_edges("llm", tools_condition)
        graph.add_edge("tools", "llm")

        return graph.compile(store=self.store)

    
    def _serialize(self, final_state: State) -> List[Dict[str, Any]]:
        serialized = []
        
        for msg in final_state["messages"]:
            msg_dict = {
                "type": msg.__class__.__name__,
                "content": getattr(msg, 'content', ''),
            }
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            serialized.append(msg_dict)

        return serialized

    
    async def run(
        self, 
        query: str,
        thread_id: str = "default"
    ) -> List[Dict[str, Any]]:

        namespace = ("conversation", thread_id)
        stored_item = self.store.get(namespace)

        if stored_item and stored_item.value:
            initial_messages = stored_item.value.get("messages", [])
        else:
            initial_messages = []
        
        initial_messages.append(HumanMessage(content=query))
        initial_state = {"messages": initial_messages}
        
        final_state = await self.graph.ainvoke(initial_state)

        self.store.put(namespace, final_state)
        
        return self._serialize(final_state)

    
    def get_conversation(self, thread_id: str = "default") -> List[Dict[str, Any]]:

        namespace = ("conversation", thread_id)
        stored_item = self.store.get(namespace)
        
        if not stored_item or not stored_item.value:
            return []
        
        return self._serialize(stored_item.value)
    

    def list_conversations(self) -> List[str]:

        results = []
        for item in self.store.search(prefix=("conversation",)):

            if len(item.namespace) == 2:
                thread_id = item.namespace[1]
                results.append(thread_id)

        return results
