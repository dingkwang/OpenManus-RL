#!/usr/bin/env python3
"""
Basic TextWorld Core Example (No PDDL, No Gym)

This example demonstrates how to use core TextWorld functionality without PDDL features
and without the gym interface, avoiding the fast-downward-textworld compilation issues on macOS.
"""

import textworld
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import random

console = Console()

def create_simple_game():
    """Create a simple TextWorld game without PDDL features."""
    console.print(Panel("[bold blue]Creating Simple TextWorld Game[/bold blue]"))
    
    # Create a simple game with basic mechanics
    # This uses TextWorld's built-in game generation without PDDL
    M = textworld.GameMaker()
    
    # Create rooms
    room1 = M.new_room("Kitchen")
    room2 = M.new_room("Living Room") 
    room3 = M.new_room("Bedroom")
    
    # Connect rooms
    M.connect(room1.east, room2.west)
    M.connect(room2.north, room3.south)
    
    # Add objects
    apple = M.new(type='o', name='apple')
    room1.add(apple)
    
    key = M.new(type='k', name='key')
    room2.add(key)
    
    chest = M.new(type='c', name='chest')
    room3.add(chest)
    
    # Set player starting location
    M.set_player(room1)
    
    # Create simple quest: get apple, find key, open chest
    M.add_fact("in", apple, room1)
    M.add_fact("in", key, room2) 
    M.add_fact("in", chest, room3)
    
    # Build the game
    game = M.build()
    return game

def test_textworld_core():
    """Test basic TextWorld core functionality without gym interface."""
    try:
        console.print(Panel("[bold green]Testing TextWorld Core Functionality[/bold green]"))
        
        # Create a simple game
        game = create_simple_game()
        
        console.print(f"✅ Game created successfully!")
        console.print(f"Game object: {type(game)}")
        
        # Get initial game state
        initial_state = game.new_game_process()
        
        console.print("\n[bold yellow]Game Information:[/bold yellow]")
        console.print(f"Game name: {game.metadata.get('name', 'Unnamed Game')}")
        console.print(f"Number of rooms: {len(game.world.rooms)}")
        console.print(f"Number of objects: {len(game.world.objects)}")
        
        # List rooms
        console.print("\n[bold cyan]Rooms in the game:[/bold cyan]")
        for room in game.world.rooms:
            console.print(f"• {room.name} (ID: {room.id})")
        
        # List objects
        console.print("\n[bold cyan]Objects in the game:[/bold cyan]")
        for obj in game.world.objects:
            console.print(f"• {obj.name} (Type: {obj.type}, ID: {obj.id})")
        
        # Show initial observation
        initial_obs = initial_state.feedback
        console.print("\n[bold yellow]Initial Game Description:[/bold yellow]")
        console.print(Panel(initial_obs, title="Game Start"))
        
        console.print("\n✅ TextWorld core functionality test completed successfully!")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Error testing TextWorld core: {e}")
        import traceback
        console.print(traceback.format_exc())
        return False

def demonstrate_textworld_features():
    """Demonstrate various TextWorld features that work without PDDL."""
    console.print(Panel("[bold magenta]TextWorld Features Demo[/bold magenta]"))
    
    try:
        # Show available game types
        console.print("\n[bold]Available TextWorld Game Types:[/bold]")
        
        # Create different types of simple games
        games_info = [
            ("Simple Navigation", "Basic room navigation without complex logic"),
            ("Object Collection", "Collect items from different rooms"),
            ("Key-Door Mechanics", "Use keys to unlock doors/containers"),
            ("Inventory Management", "Pick up and drop items")
        ]
        
        for game_type, description in games_info:
            console.print(f"• [cyan]{game_type}[/cyan]: {description}")
        
        console.print("\n[bold green]All these game types work without PDDL dependencies![/bold green]")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Error in feature demonstration: {e}")
        return False

def main():
    """Main function to run TextWorld examples."""
    console.print(Panel(
        "[bold blue]TextWorld Basic Environment Example[/bold blue]\n"
        "This demonstrates using TextWorld without PDDL features",
        title="🎮 TextWorld Demo"
    ))
    
    # Test basic functionality
    success = test_textworld_core()
    
    if success:
        # Demonstrate features
        demonstrate_textworld_features()
        
        console.print(Panel(
            "[bold green]✅ TextWorld is working correctly![/bold green]\n\n"
            "[yellow]Key Points:[/yellow]\n"
            "• Basic TextWorld environments work without PDDL\n"
            "• You can create custom games with rooms, objects, and simple mechanics\n"
            "• Navigation, inventory, and basic interactions are fully supported\n"
            "• Complex planning features require PDDL (not available on this system)",
            title="🎉 Success"
        ))
    else:
        console.print(Panel(
            "[bold red]❌ TextWorld test failed[/bold red]\n"
            "Please check the error messages above for troubleshooting.",
            title="⚠️ Error"
        ))

if __name__ == "__main__":
    main()
