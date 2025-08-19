# --------------------- ALFWorld --------------------- #
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observaitons and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_OPENMANUS_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.

Your thinking process MUST be enclosed within <think> </think> tags and follow this structure:

<think>
plan
Analyze the current situation and devise a plan to accomplish the task: {task_description}
What are the key steps needed to complete this task?

<reflection>
[Only needed when step_count > 0]
Last observation analysis: Have we made progress toward solving the task?
What did the last action accomplish? Was it successful or did it encounter any issues?
Are we closer to completing the task?
</reflection>

<memory analysis>
RAG-style retrieval from history:

[Thinking history - cite specific past reasoning from previous steps]
Example: "At step 3, I reasoned that we needed to find a knife first before attempting to slice..."
Example: "In step 5's thinking, I identified that the fridge typically contains perishable items..."

[Observation/Action history - cite specific observations and outcomes]
Example: "Step 2 observation: 'You are in the kitchen. You see a countertop 1, a cabinet 1...' - this revealed the kitchen layout"
Example: "Step 4 action 'go to fridge 1' succeeded and revealed tomato, lettuce..."
Example: "Step 6 failed with 'Nothing happens' when trying to take knife from drawer 2"

[Milestone tracking]
- Completed: Found target object at step X, Successfully picked up item at step Y
- Current state: Holding [items], Located at [location]
</memory analysis>

Future plan and next action decision:
- Immediate next step: What specific action should we take now?
- How does this action advance our plan toward completing the task?
</think>

Once you've finished your reasoning, present your chosen action within <action> </action> tags in this format:

<action>
action_choice: [selected admissible action from the list]
action_parameters: {{relevant details about the action if applicable}}
</action>
"""