# LLM

from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o")

# Tools

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def web_search(query: str) -> str:
    """Search the web for information."""
    return (
        "Here are the headcounts for each of the FAANG companies in 2024:\n"
        "1. **Facebook (Meta)**: 67,317 employees.\n"
        "2. **Apple**: 164,000 employees.\n"
        "3. **Amazon**: 1,551,000 employees.\n"
        "4. **Netflix**: 14,000 employees.\n"
        "5. **Google (Alphabet)**: 181,269 employees."
    )

# Agents

from langgraph.prebuilt import create_react_agent
math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert", # descriptive name 
    prompt="You are a math expert. Always use one tool at a time."
)

research_agent = create_react_agent(
    model=model,
    tools=[web_search],
    name="research_expert",
    prompt="You are a world class researcher with access to web search. Do not do any math."
)

# Supervisor

from langgraph_supervisor import create_supervisor

workflow = create_supervisor(
    [research_agent, math_agent], # Nodes
    model=model, # Model 
    output_mode="last_message", # What we pass back from agent to supervisor if using Orchestrator
    prompt="""You are a team supervisor managing a research expert and a math expert.
              For current events, use research_agent
              For math problems, use math_agent""",
)

# Compile and run
graph = workflow.compile()
