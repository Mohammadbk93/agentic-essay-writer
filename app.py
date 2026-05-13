import os
import uuid
from typing import TypedDict, List
from dotenv import load_dotenv, find_dotenv

from fpdf import FPDF
from io import BytesIO

import streamlit as st
from pydantic import BaseModel
from tavily import TavilyClient

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(find_dotenv(), override=True)

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing.")
    st.stop()

if not TAVILY_API_KEY:
    st.error("TAVILY_API_KEY is missing.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# -----------------------------
# Streamlit setup
# -----------------------------

st.set_page_config(
    page_title="Agentic Essay Writer",
    page_icon="🧠",
    layout="wide"
)



# -----------------------------
# Secrets
# -----------------------------


# -----------------------------
# State
# -----------------------------

class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int


class Queries(BaseModel):
    queries: List[str]


# -----------------------------
# Prompts
# -----------------------------

PLAN_PROMPT = """
You are an expert writer tasked with writing a high level outline of an essay.
Write such an outline for the user provided topic.
Give an outline of the essay along with any relevant notes or instructions for the sections.
"""

RESEARCH_PLAN_PROMPT = """
You are a researcher charged with providing information that can be used when writing the following essay.
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max.

You are a researcher.
Generate up to 3 search queries that can help gather information for this essay.
Return only useful search queries.
"""

WRITER_PROMPT = """
You are an essay assistant tasked with writing excellent 5-paragraph essays.
Generate the best essay possible for the user's request and the initial outline.
If the user provides critique, respond with a revised version of your previous attempts.
Utilize all the information below as needed: 

------

{content}
"""

REFLECTION_PROMPT = """
You are a teacher grading an essay submission.
Generate critique and recommendations for the user's submission.
Provide detailed recommendations, including requests for length, depth, style, etc.
"""

RESEARCH_CRITIQUE_PROMPT = """
You are a researcher charged with providing information that can 
be used when making any requested revisions (as outlined below).
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max.
"""


# -----------------------------
# Clients
# -----------------------------

@st.cache_resource
def get_model():
    return ChatOpenAI(model="gpt-5.1", temperature=0)


@st.cache_resource
def get_tavily():
    return TavilyClient(api_key=TAVILY_API_KEY)


model = get_model()
tavily = get_tavily()


# -----------------------------
# Nodes
# -----------------------------

def plan_node(state: AgentState):
    response = model.invoke([
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=state["task"])
    ])
    return {"plan": response.content}


def research_plan_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state["task"])
    ])

    content = state.get("content", [])

    for query in queries.queries[:3]:
        try:
            response = tavily.search(query=query, max_results=2)
            for result in response.get("results", []):
                text = result.get("content")
                url = result.get("url", "")
                title = result.get("title", "")
                if text:
                    content.append(f"Title: {title}\nURL: {url}\nContent: {text}")
        except Exception as e:
            content.append(f"Search failed for query: {query}. Error: {e}")

    return {"content": content}


def generation_node(state: AgentState):
    content = "\n\n".join(state.get("content", []))

    critique_text = ""
    if state.get("critique"):
        critique_text = f"\n\nPrevious critique:\n{state['critique']}"

    user_message = HumanMessage(
        content=f"""
Topic:
{state["task"]}

Plan:
{state["plan"]}
{critique_text}
"""
    )

    response = model.invoke([
        SystemMessage(content=WRITER_PROMPT.format(content=content)),
        user_message
    ])

    return {
        "draft": response.content,
        "revision_number": state.get("revision_number", 0) + 1
    }


def reflection_node(state: AgentState):
    response = model.invoke([
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state["draft"])
    ])
    return {"critique": response.content}


def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state["critique"])
    ])

    content = state.get("content", [])

    for query in queries.queries[:3]:
        try:
            response = tavily.search(query=query, max_results=2)
            for result in response.get("results", []):
                text = result.get("content")
                url = result.get("url", "")
                title = result.get("title", "")
                if text:
                    item = f"Title: {title}\nURL: {url}\nContent: {text}"
                    if item not in content:
                        content.append(item)
        except Exception as e:
            content.append(f"Search failed for query: {query}. Error: {e}")

    return {"content": content}


def should_continue(state: AgentState):
    if state["revision_number"] >= state["max_revisions"]:
        return END
    return "reflect"


# -----------------------------
# Build graph
# -----------------------------

@st.cache_resource
def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("planner", plan_node)
    builder.add_node("research_plan", research_plan_node)
    builder.add_node("generate", generation_node)
    builder.add_node("reflect", reflection_node)
    builder.add_node("research_critique", research_critique_node)

    builder.set_entry_point("planner")

    builder.add_edge("planner", "research_plan")
    builder.add_edge("research_plan", "generate")

    builder.add_conditional_edges(
        "generate",
        should_continue,
        {
            END: END,
            "reflect": "reflect"
        }
    )

    builder.add_edge("reflect", "research_critique")
    builder.add_edge("research_critique", "generate")

    return builder.compile(checkpointer=MemorySaver())


graph = build_graph()


def create_pdf(text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 16)
    pdf.multi_cell(0, 10, "Generated Essay")
    pdf.ln(5)

    pdf.set_font("Arial", "", 12)

    safe_text = text.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 8, safe_text)

    return bytes(pdf.output(dest="S"))

# -----------------------------
# UI
# -----------------------------

topic_areas = {
    "Sport": [
        "How AI is changing football performance analysis",
        "The impact of wearable technology on professional athletes",
        "Mental health challenges in elite sports",
        "The business growth of women’s sports",
        "How data analytics is transforming basketball strategy",
    ],
    "Technology": [
        "The future of AI agents in the workplace",
        "The impact of Nvidia Blackwell chips on AI infrastructure",
        "How MLOps improves real-world AI systems",
        "The rise of edge AI and smart devices",
        "Cybersecurity risks in the age of generative AI",
    ],
    "Finance and Economy": [
        "How AI is changing investment decision-making",
        "The future of digital banking",
        "Inflation and its impact on young investors",
        "The role of AI in fraud detection",
        "How automation affects the global job market",
    ],
    "Health": [
        "How AI is improving medical diagnosis",
        "The role of wearable devices in personal health",
        "Mental health support through digital tools",
        "The future of personalized medicine",
        "The benefits and risks of AI in healthcare",
    ],
    "Marketing": [
        "How AI is changing digital marketing",
        "The power of personalization in online advertising",
        "Influencer marketing and consumer trust",
        "The future of SEO in the AI era",
        "How brands use data to understand customers",
    ],
}

with st.sidebar:
    st.header("Settings")

    max_revisions = st.slider(
        "Max revisions",
        min_value=1,
        max_value=5,
        value=2
    )

    show_research = st.checkbox("Show research content", value=False)

    st.divider()

    st.subheader("Explore topic areas")
    st.write("Choose an area, then click a topic to use it instantly.")

    for area, topics in topic_areas.items():
        with st.expander(area):
            for topic_option in topics:
                if st.button(topic_option, key=f"{area}_{topic_option}", use_container_width=True):
                    st.session_state["selected_topic"] = topic_option



st.title("🧠 Agentic Essay Writer")

st.write(
    "Turn any idea into a polished essay with AI-powered planning, web research, feedback, and revision."
)

default_topic = st.session_state.get("selected_topic", "")

task = st.text_input(
    "Essay topic",
    value=default_topic,
    placeholder="Type something..."
)

run_button = st.button("Generate Essay", type="primary")

if run_button:
    if not task.strip():
        st.warning("Please enter a topic.")
        st.stop()

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "task": task,
        "max_revisions": max_revisions,
        "revision_number": 0,
        "content": []
    }

    final_state = None

    with st.spinner("Generating your essay..."):
        for event in graph.stream(initial_state, config):
            final_state = event
            node_name = list(event.keys())[0]
            st.write(f"✅ Completed: `{node_name}`")

    st.success("Essay generated successfully!")

    final_data = list(final_state.values())[0]
    final_essay = final_data.get("draft", "No draft generated.")

    st.subheader("Final Essay")
    st.write(final_essay)

    pdf_bytes = create_pdf(final_essay)

    st.download_button(
        label="Download Essay as PDF",
        data=pdf_bytes,
        file_name="generated_essay.pdf",
        mime="application/pdf"
    )

    with st.expander("Essay Plan"):
        st.write(final_data.get("plan", "No plan generated."))

    with st.expander("Latest Critique"):
        st.write(final_data.get("critique", "No critique generated."))

    if show_research:
        with st.expander("Research Content"):
            for i, item in enumerate(final_data.get("content", []), start=1):
                st.markdown(f"### Source {i}")
                st.write(item)