"""Formats everything the other agents produced into the user-facing answer."""

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.prompts import FINAL_AGENT_PROMPT, FINAL_SYSTEM_PROMPT
from src.clients.llm import get_llm
from src.graph.state import TravelState
from src.utils.async_utils import bump_llm_calls
from src.utils.async_utils import truncate

def final_agent(state: TravelState):
    # --- ADD TRUNCATION HERE BEFORE PROMPT FORMATTING ---
    flight_summary = truncate(state.get("flight_results", ""), 1500)
    hotel_summary = truncate(state.get("hotel_results", ""), 1500)
    weather_summary = truncate(state.get("weather_results", ""), 1000)
    itinerary_summary = truncate(state.get("itinerary", ""), 2000)

    prompt = FINAL_AGENT_PROMPT.format(
        user_query=state["user_query"],
        flight_results=flight_summary,
        hotel_results=hotel_summary,
        weather_results=weather_summary,
        itinerary=itinerary_summary,
    )

    response = get_llm().invoke([
        SystemMessage(content=FINAL_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    return {
        "messages": [response],
        "llm_calls": bump_llm_calls(state),
    }
