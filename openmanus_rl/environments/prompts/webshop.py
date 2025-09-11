# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --------------------- WebShop --------------------- #
WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert agent operating in the WebShop e‑commerce environment.
Your task is: {task_description}
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [
{available_actions}
].

Please begin by analyzing the situation and planning your approach:

<plan>
Analyze the current shopping situation and devise a plan to accomplish the task: {task_description}
What are the key steps needed to complete this task (e.g., search with the right keywords, open a relevant product, compare options, select attributes, finalize)?
Based on the current observation, what should be my immediate next step?
How does this action advance my plan toward completing the shopping goal?
</plan>

Finally, choose ONE admissible action for the current step and present it within <action> </action> tags.
"""

WEBSHOP_TEMPLATE = """
You are an expert agent operating in the WebShop e‑commerce environment.
Your task is: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [
{available_actions}
].

Now it's your turn to take an action.

You should first recall relevant past experience and reason from our conversation history, then MUST summarize within <memory_recall> </memory_recall> tags like this:

<memory_recall>
[Recall relevant past experiences and reason from our conversation history]
Recent action history ({step_count} steps taken): {action_history}
- What similar shopping situations have I encountered?
- What strategies worked or failed before (e.g., search terms, product filtering, option selection)?
- What products, attributes, or pages have I already explored?
- What was my previous reasoning and plan?
</memory_recall>

After that, you should reflect on the last action and its outcome, then MUST summarize within <reflection> </reflection> tags like this:

<reflection>
[Reflect on the last action and its outcome]
- What did my last action accomplish?
- Was it successful or did it encounter issues?
- How does this outcome affect my plan?
- Am I making progress toward the task goal: {task_description}?
</reflection>

After that, you should plan the next step based on memory and reflection, then MUST summarize within <think> </think> tags like this:

<think>
[Plan the next step based on memory and reflection]
- Given what I've learned, what should I do next?
- How does this action fit into my overall shopping strategy?
- What do I expect this action to achieve now?
</think>

Finally, choose ONE admissible action for the current step and present it within <action> </action> tags.
"""
