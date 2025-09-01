# --------------------- Tool Use --------------------- #

TOOL_USE_TEMPLATE_NO_HIS = """
You are an expert research assistant capable of using various tools to gather information and solve complex problems.

Task: {task_description}

Available Tools:
{available_tools}

Current Observation: {current_observation}

Instructions:
1. Analyze the task and determine what information you need
2. Use available tools to gather information when needed
3. Reason through the information step by step  
4. When you have sufficient information, provide your final answer in <answer></answer> tags

Format for tool usage:
<tool_call>
tool: [tool_name]
parameters: {{"param1": "value1", "param2": "value2"}}
</tool_call>

Now it's your turn to take an action. You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should either use a tool or provide your final answer within <answer> </answer> tags.
"""

TOOL_USE_TEMPLATE = """
You are an expert research assistant capable of using various tools to gather information and solve complex problems.

Task: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}

You are now at step {current_step}.
Current Observation: {current_observation}

Available Tools:
{available_tools}

Instructions:
1. Consider the current observation and your progress so far
2. Determine if you need more information or if you can provide a final answer  
3. Use tools if you need additional information
4. Provide your final answer in <answer></answer> tags when ready

Format for tool usage:
<tool_call>
tool: [tool_name]  
parameters: {{"param1": "value1", "param2": "value2"}}
</tool_call>

Now it's your turn to take an action. You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should either use a tool or provide your final answer within <answer> </answer> tags.
"""
