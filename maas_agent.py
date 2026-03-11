import os
import pandas as pd
import pickle
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

# Load the trained model artifact
MODEL_PATH = 'taxi_demand_model.pkl'
with open(MODEL_PATH, 'rb') as f:
    demand_model = pickle.load(f)

@tool
def predict_taxi_demand(location_id: int, hour: int, day_of_week: int, is_weekend: int, is_rush_hour: int) -> str:
    """
    Predicts ride-sharing demand for a specific region and time.
    Call this tool when forecasting vehicle dispatch or demand.
    
    Args:
        location_id (int): Zone ID (e.g., 1).
        hour (int): Hour of the day (0-23).
        day_of_week (int): Day of the week (0=Mon, 6=Sun).
        is_weekend (int): 1 if weekend, 0 otherwise.
        is_rush_hour (int): 1 if weekday 07:00-09:00 or 17:00-19:00, 0 otherwise.
    """
    input_data = pd.DataFrame({
        'PULocationID': [location_id],
        'hour': [hour],
        'day_of_week': [day_of_week],
        'is_weekend': [is_weekend],
        'is_rush_hour': [is_rush_hour]
    })
    
    prediction = demand_model.predict(input_data)[0]
    return f"Predicted demand: {int(prediction)} vehicles."

def run_agent(query: str) -> str:
    """Initialize LangGraph Agent and execute the user query."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    tools = [predict_taxi_demand]
    
    agent_executor = create_react_agent(llm, tools)
    
    system_prompt = SystemMessage(content="""You are an AI assistant for a MaaS operation team.
    Answer based ONLY on the prediction tool's results. Do not hallucinate numbers.
    Reply professionally and concisely in Traditional Chinese (繁體中文).""")
    
    user_prompt = HumanMessage(content=query)
    
    result = agent_executor.invoke({"messages": [system_prompt, user_prompt]})
    return result["messages"][-1].content

if __name__ == "__main__":
    # Ensure API key is set in environment variables
    if "GROQ_API_KEY" not in os.environ:
        print("Please set GROQ_API_KEY environment variable.")
    else:
        test_query = "幫我算一下今天禮拜五晚上 7 點，1 號地區大概會需要調度幾台車？"
        response = run_agent(test_query)
        print("Agent Response:\n", response)