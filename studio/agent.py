# Load model and API Keys
import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

tool_llm = ChatOllama(model="llama3.2:latest", temperature=0.9) # Using llama 3.2 as a tool calling router 

SARVAM_API_KEY = os.getenv("Sarvam-API")

#Using Sarvam M for conversation as it has native support of indic Languages 
Convo_llm = ChatOpenAI(model ="sarvam-m",
                 api_key=SARVAM_API_KEY,
                 base_url="https://api.sarvam.ai/v1",
                 ##model_kwargs={"reasoning_effort": "medium"}
                 )


from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from langchain.messages import SystemMessage

#Graph State 
class agent_state(MessagesState):
    language : str = Field(description="Preffered Language of the user") 

def conversation_agent(state:agent_state):
    current_language = state.get("language","unknown") 

    convo_sys_prompt = f'''
    You are Retail Saarthi, a knowledgeable and friendly digital assistant for Indian Kirana store owners.

    Your Task: > 
    1. Respond to the user's query regarding their business, inventory, or sales.
    2. Use the language specified: {current_language}.

    Language Guidelines:

    [STRICT] If the language is 'unknown' or not clearly mentioned, politely ask the user a question to know which langauge they wish to communicate with ,in both English and Hindi which language they prefer (e.g., Hindi, Telugu, Kannada, etc.).

    Use professional yet accessible simple retail terminology (e.g., using terms like 'stock', 'profit', or 'khata').

    If the language is provided, do NOT switch to English unless technical terms require it.
    
    '''

    response = Convo_llm.invoke([SystemMessage(content=convo_sys_prompt)]+state["messages"])

    return {"messages": [response]}


#Simple graph to test the model's responses
from langgraph.graph import START,StateGraph,MessagesState,END

from langgraph.prebuilt import ToolNode,tools_condition
from IPython.display import Image, display

def Retail_saarthi(state:MessagesState):
    return {"messages": [llm_with_tools.invoke([SystemMessage(content=system_prompt)]+state["messages"])]}


builder = StateGraph(agent_state)

builder.add_node("Agent",conversation_agent)
#builder.add_node("tools",ToolNode(tools))

builder.add_edge(START,"Agent")
#builder.add_conditional_edges("Agent",tools_condition)
#builder.add_edge("tools","Agent")
builder.add_edge("Agent",END)

react_graph = builder.compile()