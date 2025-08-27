#!/usr/bin/env python3
import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import requests

# Configure project imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openmanus_rl.multi_turn_rollout.openmanus_rollout import OpenmanusRollout
from openmanus_rl.environments.env_manager import make_envs
from openmanus_rl.environments.prompts.alfworld import ALFWORLD_OPENMANUS_TEMPLATE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Experiment configuration with sensible defaults."""
    batch_size: int = 1
    max_steps: int = 10
    seed: int = 42
    save_trajectories: bool = True
    output_dir: str = "trajectories"
    history_window: int = 3
    
    @property
    def env_config(self):
        return {
            'env_name': 'alfworld/AlfredTWEnv',
            'seed': self.seed,
            'max_steps': self.max_steps,
            'history_length': self.history_window,
            'rollout': type('RolloutConfig', (), {'n': 0})()
        }


class TrajectoryStep:
    """Single step in a trajectory with full state information."""
    
    def __init__(self, step_num: int):
        self.step = step_num
        self.observation_before = None
        self.admissible_actions = []
        self.llm_prompt = None
        self.llm_response = None
        self.parsed_action = None
        self.reward = 0.0
        self.done = False
        self.won = False
        self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'step': self.step,
            'state': {
                'observation': self.observation_before,
                'admissible_actions': self.admissible_actions
            },
            'agent_output': {
                'raw_response': self.llm_response,
                'action': self.parsed_action
            },
            'transition': {
                'reward': self.reward,
                'done': self.done
            },
            'metadata': self.metadata
        }


class LLMAgent:
    """Agent that interfaces with LLM APIs for action generation."""
    
    def __init__(self):
        # Check environment for API credentials
        self._setup_api()
        self.history = []
        self.current_task = None
        self.step_counter = 0
        
    def _setup_api(self):
        """Configure API based on environment variables."""
        self.api_key = os.getenv('OAI_KEY')
        self.api_endpoint = os.getenv('OAI_ENDPOINT')
        
        if self.api_key and self.api_endpoint:
            self.api_enabled = True
            logger.info(f"API configured: {self.api_endpoint[:30]}...")
        else:
            self.api_enabled = False
            logger.warning("No API credentials found, using heuristic fallback")
    
    def reset(self, task_description: str):
        """Reset agent state for new episode."""
        self.history.clear()
        self.current_task = task_description
        self.step_counter = 0
    
    def act(self, observation: str, admissible_actions: List[str]) -> Tuple[str, str]:
        """
        Generate action based on current observation.
        
        Returns:
            Tuple of (raw_response, action)
        """
        self.step_counter += 1
        
        # Build context from recent history
        context = self._build_context()
        
        # Generate prompt using template
        prompt = self._create_prompt(observation, admissible_actions, context)
        
        # Get response from LLM or fallback
        if self.api_enabled:
            response = self._query_llm(prompt)
        else:
            response = self._heuristic_action(admissible_actions)
        
        # Update history
        self.history.append({
            'step': self.step_counter,
            'observation': observation[:200],  # Truncate for memory
            'response': response
        })
        
        # Keep history bounded
        if len(self.history) > 5:
            self.history.pop(0)
        
        return response, self._extract_action(response)
    
    def _build_context(self) -> str:
        """Build context string from recent history."""
        if not self.history:
            return "No previous actions taken."
        
        context_parts = []
        for entry in self.history[-3:]:  # Last 3 steps
            obs_snippet = entry['observation'][:100]
            context_parts.append(f"Step {entry['step']}: {obs_snippet}...")
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, observation: str, actions: List[str], context: str) -> str:
        """Format prompt using the template."""
        return ALFWORLD_OPENMANUS_TEMPLATE.format(
            task_description=self.current_task or "Complete the task",
            step_count=max(0, self.step_counter - 1),
            history_length=min(3, len(self.history)),
            action_history=context,
            current_step=self.step_counter,
            current_observation=observation,
            admissible_actions=", ".join(actions) if actions else "none available"
        )
    
    def _query_llm(self, prompt: str) -> str:
        """Query the LLM API."""
        try:
            headers = {
                "api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            # Azure OpenAI format
            url = f"{self.api_endpoint}/openai/deployments/gpt-4o/chat/completions?api-version=2024-05-13"
            
            payload = {
                "messages": [
                    {"role": "system", "content": "You are an expert AI agent solving household tasks."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                logger.debug(f"LLM response received: {len(content)} chars")
                
                # Check if response was truncated (missing action tags)
                if '<think>' in content and not ('</action>' in content or '</reflect' in content.lower()):
                    logger.warning(f"Response appears truncated, missing action tags")
                    # Could implement retry with simpler prompt here
                
                return content
            else:
                logger.error(f"API error {response.status_code}: {response.text[:200]}")
                return self._heuristic_action([])
                
        except Exception as e:
            logger.error(f"API exception: {e}")
            return self._heuristic_action([])
    
    def _heuristic_action(self, available_actions: List[str]) -> str:
        """Simple heuristic for action selection when API unavailable."""
        # Basic exploration strategy
        action_sequence = ["look", "inventory", "go to kitchen", "go to cabinet 1", 
                          "open cabinet 1", "take mug 1", "go to sinkbasin 1",
                          "clean mug 1", "go to coffeemachine 1", "put mug 1"]
        
        idx = (self.step_counter - 1) % len(action_sequence)
        action = action_sequence[idx]
        
        # Check if action is valid
        if available_actions and action not in str(available_actions):
            # Try to find a similar valid action
            for act in available_actions:
                if any(keyword in act.lower() for keyword in ['go', 'take', 'put', 'open']):
                    action = act
                    break
        
        return f"<think>\nExploring environment systematically.\n</think>\n\n<action>\n{action}\n</action>"
    
    def _extract_action(self, response: str) -> str:
        """Extract action from structured response."""
        if '<action>' in response and '</action>' in response:
            start = response.find('<action>') + 8
            end = response.find('</action>')
            action_text = response[start:end].strip()
            
            # Handle different action formats
            if 'action_choice:' in action_text:
                parts = action_text.split('action_choice:')
                if len(parts) > 1:
                    return parts[1].split('\n')[0].strip()
            
            # Return first line if no special format
            return action_text.split('\n')[0].strip()
        
        # Smarter fallback: try to extract meaningful action from response
        response_lower = response.lower()
        
        # Look for common action patterns in the thinking
        if 'go to cabinet' in response_lower:
            # Extract cabinet number
            import re
            match = re.search(r'go to cabinet (\d+)', response_lower)
            if match:
                return f"go to cabinet {match.group(1)}"
        
        if 'open cabinet' in response_lower:
            match = re.search(r'open cabinet (\d+)', response_lower)
            if match:
                return f"open cabinet {match.group(1)}"
                
        if 'go to drawer' in response_lower:
            match = re.search(r'go to drawer (\d+)', response_lower)
            if match:
                return f"go to drawer {match.group(1)}"
        
        # Default fallback
        return "look"


class TrajectoryCollector:
    """Manages trajectory collection and storage."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.trajectories = []
        self._setup_output_dir()
    
    def _setup_output_dir(self):
        """Create output directory if needed."""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def collect(self, env, agent, rollout_processor) -> Dict[str, Any]:
        """
        Collect a single trajectory.
        
        Returns:
            Dictionary containing the full trajectory data.
        """
        trajectory = []
        obs, _ = env.reset()
        
        # Initialize agent with task
        task_description = obs['text'][0]
        agent.reset(task_description)
        
        logger.info(f"Starting trajectory collection for task: {task_description[:100]}...")
        
        for step_num in range(self.config.max_steps):
            # Create step record
            step = TrajectoryStep(step_num + 1)
            step.observation_before = obs['text'][0]
            step.admissible_actions = obs.get('admissible_actions', [None])[0] or []
            
            # Generate action
            raw_response, _ = agent.act(step.observation_before, step.admissible_actions)
            step.llm_response = raw_response
            
            # Process response through rollout system
            action, _ = rollout_processor.process_response(
                raw_response,
                episode_id=f"ep_{datetime.now().strftime('%H%M%S')}",
                step_id=step_num
            )
            step.parsed_action = action or "look"
            
            # Validate action before execution
            if step.admissible_actions and step.parsed_action not in step.admissible_actions:
                logger.warning(f"Invalid action '{step.parsed_action}', using 'look' instead")
                step.parsed_action = "look"
            
            # Execute in environment
            next_obs, rewards, dones, infos = env.step([step.parsed_action])
            
            step.reward = float(rewards[0])
            step.done = bool(dones[0])
            # Convert any numpy arrays in info to lists for JSON serialization
            info_dict = infos[0] if infos else {}
            
            # Extract admissible actions from info if not already set
            if not step.admissible_actions and 'admissible_commands' in info_dict:
                step.admissible_actions = info_dict['admissible_commands']
            
            # Store metadata excluding admissible_commands (to avoid duplication)
            step.metadata = {
                'info': {k: v.tolist() if hasattr(v, 'tolist') else v 
                        for k, v in info_dict.items() 
                        if k != 'admissible_commands'}
            }
            
            # Store won status for success determination
            step.won = info_dict.get('won', False)
            
            trajectory.append(step)
            
            # Check termination - success or environment done
            if step.done:
                logger.info(f"Episode completed at step {step_num + 1}")
                break
            elif step.won:
                logger.info(f"Task completed successfully at step {step_num + 1}!")
                break
            
            obs = next_obs
        
        return {
            'task': task_description,
            'steps': [s.to_dict() for s in trajectory],
            'total_reward': sum(s.reward for s in trajectory),
            'success': any(s.won for s in trajectory),  # True if any step shows won=True
            'length': len(trajectory)
        }
    
    def save(self, trajectory: Dict[str, Any], run_id: str = None) -> str:
        """Save trajectory to JSON file."""
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = Path(self.config.output_dir) / f"traj_{run_id}.json"
        
        # Add metadata
        output = {
            'metadata': {
                'timestamp': run_id,
                'config': asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else vars(self.config),
                'environment': 'alfworld',
                'version': '1.0'
            },
            'trajectory': trajectory
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        file_size_kb = os.path.getsize(filename) / 1024
        logger.info(f"Saved trajectory to {filename} ({file_size_kb:.2f} KB)")
        
        return str(filename)


def run_experiment(config: Optional[ExperimentConfig] = None) -> bool:
    """
    Run a complete trajectory collection experiment.
    
    Args:
        config: Experiment configuration (uses defaults if None)
    
    Returns:
        Success status
    """
    if config is None:
        config = ExperimentConfig()
    
    logger.info("Starting AlfWorld trajectory collection")
    logger.info(f"Configuration: batch_size={config.batch_size}, max_steps={config.max_steps}")
    
    try:
        # Initialize environment
        logger.info("Initializing environment...")
        
        # Create minimal config for environment
        env_config = type('Config', (), {
            'env': type('EnvConfig', (), config.env_config)(),
            'data': type('DataConfig', (), {
                'train_batch_size': config.batch_size,
                'val_batch_size': 1
            })()
        })()
        
        envs, _ = make_envs(env_config)
        
        # Initialize components
        agent = LLMAgent()
        collector = TrajectoryCollector(config)
        
        # Simple tokenizer stub
        tokenizer = type('Tokenizer', (), {'pad_token_id': 0})()
        rollout = OpenmanusRollout(env_config, tokenizer, None)
        
        # Collect trajectory
        trajectory = collector.collect(envs, agent, rollout)
        
        # Save results
        if config.save_trajectories:
            saved_path = collector.save(trajectory)
            logger.info(f"Experiment complete. Results saved to {saved_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("TRAJECTORY COLLECTION SUMMARY")
        print("="*50)
        print(f"Task: {trajectory['task'][:80]}...")
        print(f"Steps taken: {trajectory['length']}")
        print(f"Total reward: {trajectory['total_reward']:.2f}")
        print(f"Success: {'Yes' if trajectory['success'] else 'No'}")
        
        return True
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        try:
            envs.close()
        except:
            pass


if __name__ == "__main__":
    # Parse command line args if needed
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect AlfWorld trajectories')
    parser.add_argument('--steps', type=int, default=10, help='Max steps per episode')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--num_tasks', type=int, default=1, help='Number of tasks to run')
    parser.add_argument('--no-save', action='store_true', help='Disable trajectory saving')
    
    args = parser.parse_args()
    
    # Configure experiment
    exp_config = ExperimentConfig(
        max_steps=args.steps,
        batch_size=args.batch,
        save_trajectories=not args.no_save
    )
    
    # Run multiple tasks if requested
    successes = 0
    for task_idx in range(args.num_tasks):
        logger.info(f"\n=== Running task {task_idx + 1}/{args.num_tasks} ===")
        if run_experiment(exp_config):
            successes += 1
    
    logger.info(f"\n=== Completed {successes}/{args.num_tasks} tasks successfully ===")
    sys.exit(0 if successes == args.num_tasks else 1)