# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.29] - 2025-09-23

### Changed

- Updated import statements to use the new LangChain v1.0 paths
  - Replaced `create_react_agent` with `create_agent` from `langchain.agents`
  - Now using direct imports from `langchain.agents` instead of older paths
  - Updated agent creation code to match new LangChain v1.0 API

### Updated

- Dependencies in pyproject.toml
  - Set `langchain>=1.0.0a7`
  - Set `langgraph>=1.0.0a3`

### Fixed

- Example code in README.md to use the new agent creation pattern
- Tests updated to use new agent creation methods

### Notes

- There is a deprecation warning about `AgentStatePydantic` which comes from LangGraph's internal usage. This will be resolved in LangGraph v2.0 and doesn't affect functionality.

### Migration Guide

If you're updating from a previous version:

1. Update your imports:

   ```python
   # Old
   from langchain.agents import create_react_agent
   
   # New
   from langchain.agents import create_agent
   ```

2. Update agent creation calls:

   ```python
   # Old
   agent = create_react_agent(
       model=model,
       tools=tools,
       name="agent_name"
   )
   
   # New
   agent = create_agent(
       model=model,
       tools=tools,
       name="agent_name"
   )
   ```

3. Update your dependencies:

   ```toml
   langchain = ">=1.0.0a7"
   langgraph = ">=1.0.0a3"
   ```
