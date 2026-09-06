# TODO: implement in a later step
# Purpose: Self-implemented ReAct (Reasoning + Acting) loop.
# Drives the Thought/Action/Observation cycle using Gemini native function
# calling (no LangChain AgentExecutor, no agent framework). Terminates when
# the model returns a plain-text response (no function call) or when
# MAX_AGENT_STEPS is reached.
