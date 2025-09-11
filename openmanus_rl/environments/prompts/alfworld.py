# # --------------------- ALFWorld --------------------- #
# ALFWORLD_TEMPLATE_NO_HIS = """
# You are an expert agent operating in the ALFRED Embodied Environment.
# Your current observation is: {current_observation}
# Your admissible actions of the current situation are: [{admissible_actions}].

# Now it's your turn to take an action.
# You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <plan> </plan> tags. 
# Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
# """

ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your task is: {task_description}
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Please begin by analyzing the situation and planning your approach:

<plan>
Analyze the current situation and devise a plan to accomplish the task:
What are the key steps needed to complete this task?
How to advance our plan toward completing the task in immediate next step?
Based on the current observation, what should be our immediate next step?
</plan>

Finally, choose ONE admissible action for the current step and present it within <action> </action> tags.
"""


ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observaitons and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.

You should first recall relevant past experiences and reason from our conversation history, then MUST summarize within <memory_recall> </memory_recall> tags like this:

<memory_analysis>
[Recall relevant past experiences and reason from our conversation history]
- Please summarize the most relavent memory for this step.
- Please explain why this memory is helpful for the next reflection and planning.
</memory_analysis>

After that, you should reflect on the last action and its outcome, then MUST summarize within <reflection> </reflection> tags like this:

<reflection>
[Reflect on the last action and its outcome]
- What did my last action accomplish?
- Was it successful or did it encounter issues?
- How does this outcome affect my plan?
- Am I making progress toward the task goal?
</reflection>

After that, you should plan the next step based on memory and reflection, then MUST summarize within <plan> </plan> tags like this:

<plan>
[Plan the next step based on memory and reflection]
- Given what I've learned, what should I do next?
- Please explain why this plan is helpful for the next action?
- How does this action fit into my overall strategy?
- What do I expect this action to achieve?
</plan>

Finally, choose ONE admissible action for the current step and present it within <action> </action> tags.
"""
