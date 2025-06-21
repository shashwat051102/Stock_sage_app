from Agents.Reason_agent import Reason_agent
from crewai import Task
from crewai_tools import SerperDevTool



search_tool = SerperDevTool()

Reason_task = Task(
    description=(
        "A trader is seeking a detailed explanation for the predicted price of a specific stock. "
        "You will be provided with the predicted price ({predicted_actual}) and the stock symbol ({symbol}). "
        "Your task is to analyze the prediction by considering relevant market factors, recent news, "
        "company performance, and any other significant events that could impact the stock's price. "
        "Use the provided tools to gather supporting information if necessary. "
        "Clearly explain the reasoning behind the predicted price, referencing specific data or events where possible."
    ),
    expected_output=(
        "Provide a clear, point-based reason for the predicted price of the given stock symbol. "
        "Include references to market trends, news, or company fundamentals that justify the prediction."
    ),
    input_variables=["predicted_actual", "symbol"],
    tools=[search_tool],
    agent=Reason_agent,
)
