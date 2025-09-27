from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict
import psycopg

from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.postgres import PostgresStore
from langchain_core.messages import BaseMessage, messages_to_dict, messages_from_dict

from settings import Settings


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


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

    
    async def run(
        self, 
        query: str,
        thread_id: str = "default"
    ) -> List[Dict[str, Any]]:
        namespace = ("conversation", thread_id)
        stored_item = self.store.get(namespace, "state")

        if stored_item and stored_item.value:
            initial_messages = messages_from_dict(stored_item.value["messages"])
        else:
            initial_messages = []
        
        initial_messages.append(HumanMessage(content=query))
        initial_state = {"messages": initial_messages}
        
        final_state = await self.graph.ainvoke(initial_state)

        self.store.put(namespace, "state", {"messages": messages_to_dict(final_state["messages"])})
        
        return messages_to_dict(final_state["messages"])

    
    def get_conversation(self, thread_id: str = "default") -> List[Dict[str, Any]]:
        namespace = ("conversation", thread_id)
        stored_item = self.store.get(namespace, "state")
        
        if not stored_item or not stored_item.value:
            return []
        
        return messages_to_dict(messages_from_dict(stored_item.value["messages"]))
    

    def list_conversations(self) -> List[str]:
        results = []
        for item in self.store.search("conversation"):

            if len(item.namespace) == 2:
                thread_id = item.namespace[1]
                results.append(thread_id)

        return results
