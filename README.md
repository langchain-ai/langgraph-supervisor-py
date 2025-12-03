# 🤖 LangGraph Multi-Agent Supervisor

> **Note**: We now recommend using the **supervisor pattern directly via tools** rather than this library for most use cases. The tool-calling approach gives you more control over context engineering and is the recommended pattern in the [LangChain multi-agent guide](https://docs.langchain.com/oss/python/langchain/multi-agent). See our [supervisor tutorial](https://docs.langchain.com/oss/python/langchain/supervisor) for a step-by-step guide. We're making this library compatible with LangChain 1.0 to help users upgrade their existing code. If you find this library solves a problem that can't be easily addressed with the manual supervisor pattern, we'd love to hear about your use case!

A Python library for creating hierarchical multi-agent systems using [LangGraph](https://github.com/langchain-ai/langgraph). Hierarchical systems are a type of [multi-agent](https://langchain-ai.github.io/langgraph/concepts/multi_agent) architecture where specialized agents are coordinated by a central **supervisor** agent. The supervisor controls all communication flow and task delegation, making decisions about which agent to invoke based on the current context and task requirements.

## Features

- 🤖 **Create a supervisor agent** to orchestrate multiple specialized agents
- 🛠️ **Tool-based agent handoff mechanism** for communication between agents
- 📝 **Flexible message history management** for conversation control
- 🎛️ **Context engineering & propogation** for controlling information flow between agents

This library is built on top of [LangGraph](https://github.com/langchain-ai/langgraph), a powerful framework for building agent applications, and comes with out-of-box support for [streaming](https://langchain-ai.github.io/langgraph/how-tos/#streaming), [short-term and long-term memory](https://langchain-ai.github.io/langgraph/concepts/memory/) and [human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)

## Installation

```bash
pip install langgraph-supervisor
```

> [!Note]
> LangGraph Supervisor requires Python >= 3.10

## Quickstart

Here's a simple example of a supervisor managing two specialized agents:

![Supervisor Architecture](static/img/supervisor.png)

```bash
pip install langgraph-supervisor langchain-openai

export OPENAI_API_KEY=<your_api_key>
```

```python
from langchain_openai import ChatOpenAI

from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

model = ChatOpenAI(model="gpt-4o")

# Create specialized agents

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

math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    prompt="You are a math expert. Always use one tool at a time."
)

research_agent = create_react_agent(
    model=model,
    tools=[web_search],
    name="research_expert",
    prompt="You are a world class researcher with access to web search. Do not do any math."
)

# Create supervisor workflow
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    )
)

# Compile and run
app = workflow.compile()
result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "what's the combined headcount of the FAANG companies in 2024?"
        }
    ]
})
```

## Basic Message Management

You can control how messages from worker agents are added to the overall conversation history of the multi-agent system:

Include full message history from an agent:

![Full History](static/img/full_history.png)

```python
workflow = create_supervisor(
    agents=[agent1, agent2],
    output_mode="full_history"
)
```

Include only the final agent response:

![Last Message](static/img/last_message.png)

```python
workflow = create_supervisor(
    agents=[agent1, agent2],
    output_mode="last_message"
)
```

## Multi-level Hierarchies

You can create multi-level hierarchical systems by creating a supervisor that manages multiple supervisors.

```python
research_team = create_supervisor(
    [research_agent, math_agent],
    model=model,
    supervisor_name="research_supervisor"
).compile(name="research_team")

writing_team = create_supervisor(
    [writing_agent, publishing_agent],
    model=model,
    supervisor_name="writing_supervisor"
).compile(name="writing_team")

top_level_supervisor = create_supervisor(
    [research_team, writing_team],
    model=model,
    supervisor_name="top_level_supervisor"
).compile(name="top_level_supervisor")
```

## Adding Memory

You can add [short-term](https://langchain-ai.github.io/langgraph/how-tos/persistence/) and [long-term](https://langchain-ai.github.io/langgraph/how-tos/cross-thread-persistence/) [memory](https://langchain-ai.github.io/langgraph/concepts/memory/) to your supervisor multi-agent system. Since `create_supervisor()` returns an instance of `StateGraph` that needs to be compiled before use, you can directly pass a [checkpointer](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.base.BaseCheckpointSaver) or a [store](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) instance to the `.compile()` method:

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

checkpointer = InMemorySaver()
store = InMemoryStore()

model = ...
research_agent = ...
math_agent = ...

workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt="You are a team supervisor managing a research expert and a math expert.",
)

# Compile with checkpointer/store
app = workflow.compile(
    checkpointer=checkpointer,
    store=store
)
```

## Context Engineering & Advanced Message Management

A variety of powerful **context engineering** techniques are enabled through LangGraph Supervisor, allowing you to precisely control what information flows between agents. This is crucial for managing token costs, maintaining conversation relevance, and ensuring agents receive only the context they need to perform their tasks effectively.

### Why Use Custom Tools vs Direct Agent Calls?

You might wonder why not just call agents directly instead of using handoff tools. There are several important reasons:

1. **Human-in-the-Loop Support**: The supervisor architecture supports interrupts and human oversight at the orchestration level
2. **Consistent State Management**: Handoff tools ensure proper state updates and message history tracking
3. **Audit Trail**: Tool calls create a clear record of which agent was called and why
4. **Dynamic Routing**: The supervisor can choose between multiple agents based on context
5. **Error Handling**: Failed tool calls can be handled gracefully by the supervisor

### Basic Handoff Tool Customization

By default, the supervisor uses handoff tools created with the prebuilt `create_handoff_tool`. You can also create your own, custom handoff tools. Here are some ideas on how you can modify the default implementation:

* change tool name and/or description
* add tool call arguments for the LLM to populate, for example a task description for the next agent
* change what data is passed to the subagent as part of the handoff: by default `create_handoff_tool` passes **full** message history (all of the messages generated in the supervisor up to this point), as well as a tool message indicating successful handoff.

Here is an example of how to pass customized handoff tools to `create_supervisor`:

```python
from langgraph_supervisor import create_handoff_tool
workflow = create_supervisor(
    [research_agent, math_agent],
    tools=[
        create_handoff_tool(agent_name="math_expert", name="assign_to_math_expert", description="Assign task to math expert"),
        create_handoff_tool(agent_name="research_expert", name="assign_to_research_expert", description="Assign task to research expert")
    ],
    model=model,
)
```

You can also control whether the handoff tool invocation messages are added to the state. By default, they are added (`add_handoff_messages=True`), but you can disable this if you want a more concise history:

```python
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    add_handoff_messages=False
)
```

Additionally, you can customize the prefix used for the automatically generated handoff tools:

```python
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    handoff_tool_prefix="delegate_to"
)
# This will create tools named: delegate_to_research_expert, delegate_to_math_expert
```

### Advanced Context Control Patterns

For sophisticated context management, you can create custom handoff tools. Here are some useful examples:

#### 1. Summary-Based Context

Create a handoff tool that provides only a summary instead of full history:

```python
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, Send

@tool("transfer_to_subject_expert_with_summary")
def transfer_with_summary(
    task_summary: Annotated[str, "Brief summary of what the subject expert should do"],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Transfer to subject expert with only a task summary, not full history."""
    
    tool_message = ToolMessage(
        content=f"Successfully transferred to subject expert with summary",
        name="transfer_to_subject_expert_with_summary", 
        tool_call_id=tool_call_id,
    )
    
    # Create minimal context with just the summary
    summary_message = HumanMessage(content=task_summary)
    minimal_context = [summary_message]
    
    return Command(
        goto=[Send("subject_expert", {"messages": minimal_context})],
        graph=Command.PARENT,
        update={"messages": state["messages"] + [tool_message]},
    )

# Use in supervisor
workflow = create_supervisor(
    [math_agent],
    model=model,
    tools=[transfer_with_summary],  # Custom tool instead of default
    prompt="You are a supervisor. When delegating to subject expert, provide a clear task summary."
)
```

#### 2. Recent Context Window

Limit context to the most recent N messages:

```python
@tool("transfer_to_subject_expert_recent")
def transfer_with_recent_context(
    recent_message_count: Annotated[int, "Number of recent messages to include (default: 3)"] = 3,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Transfer with only recent message history."""
    
    # Get the most recent messages
    recent_messages = state["messages"][-recent_message_count:]
    
    return Command(
        goto=[Send("subject_expert", {"messages": recent_messages})],
        graph=Command.PARENT,
        update={"messages": state["messages"] + [
            ToolMessage(
                content=f"Transferred with last {len(recent_messages)} messages",
                name="transfer_to_subject_expert_recent",
                tool_call_id=tool_call_id,
            )
        ]},
    )
```

#### 3. Context Compression with pre_model_hook

For more sophisticated context management, you could use the `pre_model_hook` parameter to implement context compression at the sub-agent level:

```python
from langchain_core.messages import RemoveMessage, SystemMessage

def compress_context_hook(state):
    """Hook to compress long message histories before sending to LLM."""
    messages = state["messages"]
    
    # If history is too long, keep system message + recent messages + summary
    if len(messages) > n:
        system_msg = next((m for m in messages if m.type == "system"), None)
        recent_messages = messages[-n:]  # Keep last n messages
        
        # Create a summary of the middle messages
        middle_messages = messages[1:-n] if system_msg else messages[:-n]
        summary_content = f"Previous conversation summary: {len(middle_messages)} messages discussed various topics."
        summary_msg = SystemMessage(content=summary_content)
        
        compressed_messages = []
        if system_msg:
            compressed_messages.append(system_msg)
        compressed_messages.extend([summary_msg] + recent_messages)
        
        return {
            "messages": [RemoveMessage(id="REMOVE_ALL_MESSAGES")] + compressed_messages
        }
    
    return {"messages": messages}

# Apply to an agent
math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="subject_expert",
    pre_model_hook=compress_context_hook,  # Compress context before each LLM call
    prompt="You are a subject-matter expert. Work with the provided context efficiently."
)
```

#### 4. Role-Based Context Filtering

Filter messages based on agent roles or message types:

```python
@tool("transfer_to_specialist_filtered")
def transfer_with_filtered_context(
    include_user_messages: Annotated[bool, "Include user messages"] = True,
    include_agent_messages: Annotated[bool, "Include other agent messages"] = False,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Transfer with filtered message types."""
    
    filtered_messages = []
    for msg in state["messages"]:
        if msg.type == "human" and include_user_messages:
            filtered_messages.append(msg)
        elif msg.type == "ai" and include_agent_messages:
            filtered_messages.append(msg)
        # Always include system messages
        elif msg.type == "system":
            filtered_messages.append(msg)
    
    return Command(
        goto=[Send("specialist_agent", {"messages": filtered_messages})],
        graph=Command.PARENT,
        update={"messages": state["messages"] + [
            ToolMessage(
                content=f"Transferred with {len(filtered_messages)} filtered messages",
                name="transfer_to_specialist_filtered",
                tool_call_id=tool_call_id,
            )
        ]},
    )
```

### Custom Handoff Tool Implementation

Here is an example of what a custom handoff tool might look like:

```python
from typing import Annotated

from langchain_core.tools import tool, BaseTool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langgraph_supervisor.handoff import METADATA_KEY_HANDOFF_DESTINATION

def create_custom_handoff_tool(*, agent_name: str, name: str | None, description: str | None) -> BaseTool:

    @tool(name, description=description)
    def handoff_to_agent(
        # you can add additional tool call arguments for the LLM to populate
        # for example, you can ask the LLM to populate a task description for the next agent
        task_description: Annotated[str, "Detailed description of what the next agent should do, including all of the relevant context."],
        # you can inject the state of the agent that is calling the tool
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            name=name,
            tool_call_id=tool_call_id,
        )
        messages = state["messages"]
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            # NOTE: this is a state update that will be applied to the swarm multi-agent graph (i.e., the PARENT graph)
            update={
                "messages": messages + [tool_message],
                "active_agent": agent_name,
                # optionally pass the task description to the next agent
                # NOTE: individual agents would need to have `task_description` in their state schema
                # and would need to implement logic for how to consume it
                "task_description": task_description,
            },
        )

    handoff_to_agent.metadata = {METADATA_KEY_HANDOFF_DESTINATION: agent_name}
    return handoff_to_agent
```

### Message Forwarding

You can equip the supervisor with a tool to directly forward the last message received from a worker agent straight to the final output of the graph using `create_forward_message_tool`. This is useful when the supervisor determines that the worker's response is sufficient and doesn't require further processing or summarization by the supervisor itself. It saves tokens for the supervisor and avoids potential misrepresentation of the worker's response through paraphrasing.

```python
from langgraph_supervisor.handoff import create_forward_message_tool

# Assume research_agent and math_agent are defined as before

forwarding_tool = create_forward_message_tool("supervisor") # The argument is the name to assign to the resulting forwarded message
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    # Pass the forwarding tool along with any other custom or default handoff tools
    tools=[forwarding_tool]
)
```

This creates a tool named `forward_message` that the supervisor can invoke. The tool expects an argument `from_agent` specifying which agent's last message should be forwarded directly to the output.

## Controlling Context Shared Back Up

You can also control how much information flows back from child agents to the supervisor. This is useful for keeping the supervisor's context clean and focused.

### Output Mode Configuration

Use the `output_mode` parameter to control what gets added back to the supervisor's message history:

```python
# Include only the final response from each agent
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    output_mode="last_message"  # Default - only final response
)

# Include the full conversation history from each agent
workflow = create_supervisor(
    [research_agent, math_agent], 
    model=model,
    output_mode="full_history"  # Complete agent interaction history
)
```

### Handoff Back Messages

Control whether handoff operations themselves are included in the message history:

```python
# Disable handoff messages to keep history cleaner
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    add_handoff_messages=False,  # Don't include "Transferred to X" messages
    add_handoff_back_messages=False  # Don't include "Returned from X" messages
)
```

## Using Functional API 

LangGraph's [Functional API](https://langchain-ai.github.io/langgraph/concepts/low_level/#functional-api) provides a decorator-based approach for building agents using simple Python functions instead of explicit graph construction. This is particularly useful for creating lightweight, task-specific agents that can be easily integrated into supervisor workflows.

Here's a simple example of a supervisor managing two specialized agentic workflows created using Functional API:

```bash
pip install langgraph-supervisor langchain-openai

export OPENAI_API_KEY=<your_api_key>
```

```python
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from langchain_openai import ChatOpenAI

from langgraph.func import entrypoint, task
from langgraph.graph import add_messages

model = ChatOpenAI(model="gpt-4o")

# Create specialized agents

# Functional API - Agent 1 (Joke Generator)
@task
def generate_joke(messages):
    """First LLM call to generate initial joke"""
    system_message = {
        "role": "system", 
        "content": "Write a short joke"
    }
    msg = model.invoke(
        [system_message] + messages
    )
    return msg

@entrypoint()
def joke_agent(state):
    joke = generate_joke(state['messages']).result()
    messages = add_messages(state["messages"], [joke])
    return {"messages": messages}

joke_agent.name = "joke_agent"

# Graph API - Agent 2 (Research Expert)
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

research_agent = create_react_agent(
    model=model,
    tools=[web_search],
    name="research_expert",
    prompt="You are a world class researcher with access to web search. Do not do any math."
)

# Create supervisor workflow
workflow = create_supervisor(
    [research_agent, joke_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a research expert and a joke expert. "
        "For current events, use research_agent. "
        "For any jokes, use joke_agent."
    )
)

# Compile and run
app = workflow.compile()
result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Share a joke to relax and start vibe coding for my next project idea."
        }
    ]
})

for m in result["messages"]:
    m.pretty_print()
```