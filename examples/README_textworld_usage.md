# TextWorld Usage Guide (Without PDDL)

This guide shows how to use TextWorld environments in OpenManus-RL without PDDL features, avoiding compilation issues on macOS.

## Quick Start

Run the examples to see TextWorld in action:

```bash
# Basic functionality test
uv run python examples/simple_textworld_test.py

# Interactive demo with game creation
uv run python examples/textworld_interactive_demo.py
```

## What's Available ✅

### Core Features
- **Game Creation**: Build custom text-based environments
- **Room Navigation**: Multi-room environments with connections  
- **Object Interaction**: Pick up, drop, and manipulate objects
- **Inventory Management**: Track player inventory state
- **Action Processing**: Process natural language commands
- **Feedback Generation**: Get descriptive text observations

### Basic Game Structure
```python
import textworld

# Create a game maker
M = textworld.GameMaker()

# Create rooms
room1 = M.new_room("Starting Room")
room2 = M.new_room("Goal Room")

# Connect rooms
M.connect(room1.east, room2.west)

# Add objects
sword = M.new(type='o', name='magic sword')
room1.add(sword)

# Set player starting location
M.set_player(room1)

# Build the game
game = M.build()
```

## What's NOT Available ❌

### PDDL-Dependent Features
- **Automated Planning**: Complex rule-based planning systems
- **Advanced Logic**: Complex conditional rules and constraints
- **ALFWorld Integration**: Full ALFWorld environments (require PDDL)

### Why These Are Unavailable
The `fast-downward-textworld` package has C++ compilation issues on macOS with Apple Clang, specifically:
- Error: `no member named 'construct' in 'optional<type-parameter-0-0 &>'`
- This affects all PDDL-based planning features

## Usage for RL Development

### 1. Environment Creation
```python
import textworld

def create_rl_environment():
    M = textworld.GameMaker()
    
    # Create multi-room environment
    rooms = []
    for i in range(3):
        room = M.new_room(f"Room_{i}")
        rooms.append(room)
    
    # Connect rooms in sequence
    for i in range(len(rooms)-1):
        M.connect(rooms[i].east, rooms[i+1].west)
    
    # Add objectives (objects to collect)
    for i, room in enumerate(rooms):
        obj = M.new(type='o', name=f'item_{i}')
        room.add(obj)
    
    M.set_player(rooms[0])
    return M.build()
```

### 2. Agent Training Loop
```python
# Create environment
game = create_rl_environment()

# Training loop concept (simplified)
def train_agent(game, agent):
    for episode in range(num_episodes):
        # Reset game state
        game_state = initialize_game_state(game)
        
        while not done:
            # Agent selects action
            action = agent.select_action(observation)
            
            # Process action and get feedback
            # (You'll need to implement game state management)
            observation, reward, done = process_action(game_state, action)
            
            # Update agent
            agent.update(observation, reward)
```

### 3. Custom Reward Functions
```python
def calculate_reward(game_state, action, new_state):
    reward = 0
    
    # Reward for collecting items
    if 'take' in action and 'successfully' in new_state.feedback:
        reward += 10
    
    # Reward for reaching goal room
    if 'Goal Room' in new_state.current_room:
        reward += 50
    
    # Small penalty for each step (encourage efficiency)
    reward -= 1
    
    return reward
```

## Integration with OpenManus-RL

### Environment Wrapper
Create a wrapper to integrate TextWorld with your RL framework:

```python
class TextWorldEnvironment:
    def __init__(self, game):
        self.game = game
        self.reset()
    
    def reset(self):
        # Initialize game state
        self.current_state = self.game.new_game_process()
        return self.get_observation()
    
    def step(self, action):
        # Process action
        old_score = self.current_state.score
        self.current_state.step(action)
        
        # Calculate reward
        reward = self.current_state.score - old_score
        
        # Check if done
        done = self.current_state.done
        
        return self.get_observation(), reward, done, {}
    
    def get_observation(self):
        return self.current_state.feedback
```

## Examples Provided

1. **`simple_textworld_test.py`**: Basic functionality verification
2. **`textworld_interactive_demo.py`**: Comprehensive demo with multiple games
3. **`textworld_basic_example.py`**: Detailed example with game interaction

## Troubleshooting

### Common Issues
1. **Import errors**: Make sure `alfworld` is not installed (it forces PDDL dependencies)
2. **PDDL compilation errors**: These are expected and why we avoid PDDL features
3. **Game creation errors**: Use simple object types ('o' for objects)

### Working Configuration
- ✅ `textworld>=1.6.2` (without PDDL extra)
- ❌ `textworld[pddl]` (causes compilation issues)
- ❌ `alfworld` (depends on PDDL features)

## Next Steps

1. **Experiment** with the provided examples
2. **Create custom games** for your specific RL tasks
3. **Implement reward functions** based on your objectives
4. **Integrate** with your preferred RL framework (stable-baselines3, etc.)

For complex planning scenarios, consider:
- Using Docker with Linux for PDDL features
- Alternative text-based environments
- Custom rule systems without PDDL

## Support

If you encounter issues:
1. Check that you're not using PDDL-dependent features
2. Verify TextWorld version and dependencies
3. Run the test scripts to confirm basic functionality
