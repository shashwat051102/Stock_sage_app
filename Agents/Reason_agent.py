from crewai import Agent
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from utils.Astra_DB_search import astra_search_tool
import openai
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables.")

openai.api_key = api_key
llm = ChatOpenAI(api_key=api_key, model="gpt-4.1-mini")

search_tool = SerperDevTool()

# Add Astra Search Tool
astra_search_tool_function = astra_search_tool

Reason_agent = Agent(
    role="Stock Price Reasoning Analyst",
    goal="""
    You will receive the predicted stock price ({predicted_actual}) and the stock symbol ({symbol}) of a company.
    Your task is to conduct a thorough web search using the SerperDevTool and query Astra DB using the Astra Search Tool to identify recent news, events, financial reports, or market trends that could explain the predicted stock price movement.
    Analyze multiple credible sources, including news articles, press releases, earnings reports, analyst opinions, and macroeconomic factors.
    Present your findings in the following format:
    
    Reason for predicted stock price: {predicted_actual} for {symbol} is because of the following news and factors:
    1. [Summary of News/Event 1] - [Brief explanation of its impact]
    2. [Summary of News/Event 2] - [Brief explanation of its impact]
    3. [Summary of News/Event 3] - [Brief explanation of its impact]
    4. [Summary of News/Event 4] - [Brief explanation of its impact]
    
    After listing the reasons, provide a concise combined conclusion that synthesizes the main drivers behind the predicted stock price, highlighting the most influential factors.
    Ensure your analysis is clear, objective, and based on the most recent and relevant information available.
    """,
    backstory="""
    You are an expert stock market analyst with years of experience in financial research and news analysis.
    Your expertise lies in connecting real-world events, financial data, and market sentiment to stock price movements.
    You use advanced web search tools and critical thinking to provide detailed, evidence-based explanations for stock price predictions.
    Your insights help investors understand the underlying reasons for price changes and make informed decisions.
    """,
    tool=[search_tool, astra_search_tool_function],
    llm=llm,
)