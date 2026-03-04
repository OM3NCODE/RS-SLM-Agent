import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


# Load environment variables from .env file
load_dotenv()

SARVAM_API_KEY = os.getenv("Sarvam-API")

# Initialize the Sarvam LLM 
convo_llm = ChatOpenAI(model ="sarvam-m",
                 api_key=SARVAM_API_KEY,
                 base_url="https://api.sarvam.ai/v1",)

tool_llm = ChatOllama(model="llama3.2:latest", temperature=0.9)

#Define graph state 

from langgraph.graph import MessagesState
from langchain.messages import SystemMessage
from typing import Optional

class Agent_state(MessagesState):
    Query: Optional[str] # User query in english
    User_intent: Optional[str] # User intent in english
    language: Optional[str]
    need_tool: Optional[bool]

import json
from langchain_core.messages import AIMessage

#Nodes : 


#First conversation node - to understand user intent and decide if tool is needed.
def Retail_saarthi(Agent_state):
    sys_prompt = """Your role is to converse naturally with the shop owner, understand their core need, and translate their request for our backend systems. Always maintain a polite, helpful, and culturally appropriate tone.

    Analyze the conversation history and the user's latest message. You must output your response EXACTLY as a valid JSON object. Do not include any conversational filler before or after the JSON block. 

    Your JSON output must follow this exact structure:

    {
      "intent": "Determine the user's goal. Use ONLY one of these exact phrases: 'General Chat', 'Inventory Help', 'Demand Forecasting', or 'Policy Question'.",
      "needs_tool_call": "true or false. (Set to false if the intent is 'General Chat'. Set to true for 'Inventory Help', 'Demand Forecasting', or 'Policy Question').,
      "english_query": "Translate the user's latest query into clear, actionable English so the backend tool-caller can understand exactly what needs to be calculated or searched. If the intent is 'General Chat', write 'N/A'.",
      "chat_response": "If the intent is 'General Chat', write your full, helpful reply here in the exact language and script the user spoke. If the intent requires backend data, write a brief, polite acknowledgement in the user's language and script here (e.g., 'Let me check the stock for you...', 'जी, मैं अभी चेक करता हूँ...', etc.)."
    }

    ### EXAMPLES ###

    User: "ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಾ?"
    Output:
    {
      "intent": "General Chat",
      "needs_tool_call": "false",
      "english_query": "N/A",
      "chat_response": "ನಮಸ್ಕಾರ! ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ, ಧನ್ಯವಾದಗಳು. ಇವತ್ತು ನಿಮ್ಮ ಅಂಗಡಿಯ ಕೆಲಸದಲ್ಲಿ ನಾನು ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?"
    }

    User: "Hello, please check how many boxes of Parle-G we have left in the back room."
    Output:
    {
      "intent": "Inventory Help",
      "needs_tool_call": "true",
      "english_query": "Check current inventory stock levels for Parle-G boxes in the back room.",
      "chat_response": "Sure thing, let me quickly check the stock room numbers for Parle-G for you..."
    }

      User: "दीवाली आ रही है, अगले महीने सरसों के तेल की कितनी मांग होगी?"
      Output:
      {
        "intent": "Demand Forecasting",
        "needs_tool_call": "true",
        "english_query": "Forecast the sales demand for mustard oil for the next month, factoring in the upcoming Diwali festival.",
        "chat_response": "जी बिल्कुल, मैं अगले महीने के लिए सरसों के तेल की मांग चेक करके आपको बताता हूँ..."
      }

      User: "నమస్కారం, పాడైన చిప్స్ ప్యాకెట్లను తిరిగి ఇవ్వడానికి సప్లయర్ పాలసీ ఏమిటి?"
      Output:
      {
        "intent": "Policy Question",
        "needs_tool_call": "true",
        "english_query": "Search the supplier policy for returning damaged goods, specifically chips packets.",
        "chat_response": "ఒక్క నిమిషం అండి, పాడైన ప్యాకెట్ల కోసం నేను సప్లయర్ రిటర్న్ పాలసీని చెక్ చేస్తాను..."
      }
      """
    
    message = Agent_state["messages"][-1]

    response = convo_llm.invoke([SystemMessage(content=sys_prompt), message])

    try :
        # We clean the response to extract the JSON part and parse it.
        clean_text = response.content.strip("` \n").replace("json\n", "") 
        json_response = json.loads(clean_text)

        #Extract the values 
        intent = json_response.get("intent")
        needs_tool_call = json_response.get("needs_tool_call")
        english_query = json_response.get("english_query")
        chat_response = json_response.get("chat_response")

        return {
            "User_intent": intent,
            "need_tool": needs_tool_call,
            "Query": english_query,
            "messages": [AIMessage(content=chat_response)]
        }
    except json.JSONDecodeError:
        print("Failed to parse JSON. Raw output:", response)
        error_message = "Sorry, I had trouble understanding your request. Could you please rephrase it?"
        return {
            "User_intent": "Error",
            "need_tool": False,
            "messages": [AIMessage(content=error_message)]
        }



from langgraph.graph import START,StateGraph,MessagesState,END

from langgraph.prebuilt import ToolNode,tools_condition
from IPython.display import Image, display

builder = StateGraph(Agent_state)

builder.add_node("Agent",Retail_saarthi)

builder.add_edge(START,"Agent")
builder.add_edge("Agent",END)

react_graph = builder.compile()