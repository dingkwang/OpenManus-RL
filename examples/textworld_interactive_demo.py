#!/usr/bin/env python3
"""
TextWorld Interactive Demo

This demonstrates how to create and interact with TextWorld environments
without PDDL features, suitable for RL training and agent development.
"""

import textworld
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
import random

console = Console()

def create_treasure_hunt_game():
    """Create a treasure hunt game with multiple rooms and objectives."""
    console.print("[bold blue]Creating Treasure Hunt Game...[/bold blue]")
    
    M = textworld.GameMaker()
    
    # Create rooms
    entrance = M.new_room("Entrance Hall")
    library = M.new_room("Library")
    kitchen = M.new_room("Kitchen") 
    garden = M.new_room("Garden")
    attic = M.new_room("Attic")
    
    # Connect rooms
    M.connect(entrance.north, library.south)
    M.connect(entrance.east, kitchen.west)
    M.connect(kitchen.north, garden.south)
    M.connect(library.east, attic.west)
    
    # Add objects
    key = M.new(type='o', name='brass key')
    book = M.new(type='o', name='ancient book')
    apple = M.new(type='o', name='red apple')
    treasure = M.new(type='o', name='golden treasure')
    chest = M.new(type='o', name='wooden chest')
    
    # Place objects in rooms
    entrance.add(key)
    library.add(book)
    kitchen.add(apple)
    garden.add(chest)
    attic.add(treasure)
    
    # Set player starting location
    M.set_player(entrance)
    
    # Add some facts about the world
    M.add_fact("in", key, entrance)
    M.add_fact("in", book, library)
    M.add_fact("in", apple, kitchen)
    M.add_fact("in", chest, garden)
    M.add_fact("in", treasure, attic)
    
    game = M.build()
    console.print("✅ Treasure hunt game created!")
    return game

def create_simple_quest_game():
    """Create a simple quest game with basic objectives."""
    console.print("[bold blue]Creating Simple Quest Game...[/bold blue]")
    
    M = textworld.GameMaker()
    
    # Create a simple 3-room layout
    start_room = M.new_room("Starting Room")
    middle_room = M.new_room("Middle Room")
    end_room = M.new_room("Goal Room")
    
    # Connect rooms linearly
    M.connect(start_room.east, middle_room.west)
    M.connect(middle_room.east, end_room.west)
    
    # Add quest items
    sword = M.new(type='o', name='magic sword')
    shield = M.new(type='o', name='steel shield')
    potion = M.new(type='o', name='health potion')
    
    # Place items
    start_room.add(sword)
    middle_room.add(shield)
    end_room.add(potion)
    
    # Set player start
    M.set_player(start_room)
    
    game = M.build()
    console.print("✅ Simple quest game created!")
    return game

def demonstrate_game_interaction(game, game_name):
    """Demonstrate basic game interaction."""
    console.print(f"\n[bold yellow]🎮 Playing: {game_name}[/bold yellow]")
    
    # Create a simple game state representation
    console.print(Panel(f"Game: {game_name}\nRooms: {len(game.world.rooms)}\nObjects: {len(game.world.objects)}", 
                       title="Game Info", border_style="green"))
    
    # Show room layout
    console.print("\n[bold cyan]Game World Layout:[/bold cyan]")
    for room in game.world.rooms:
        objects_in_room = [obj.name for obj in room.content if obj.type != 'P']
        exits = [f"{direction}" for direction in room.exits.keys()]
        
        console.print(f"🏠 [yellow]{room.name}[/yellow]")
        if objects_in_room:
            console.print(f"   Objects: {', '.join(objects_in_room)}")
        if exits:
            console.print(f"   Exits: {', '.join(exits)}")
    
    # Show what a basic interaction would look like
    console.print("\n[bold cyan]Sample Game Interaction:[/bold cyan]")
    console.print("Player starts in:", game.world.player_room.name)
    
    sample_commands = [
        "look - examine current room",
        "inventory - check what you're carrying", 
        "take <object> - pick up an item",
        "go <direction> - move to another room",
        "drop <object> - put down an item"
    ]
    
    console.print("\n[dim]Available commands:[/dim]")
    for cmd in sample_commands:
        console.print(f"  • {cmd}")
    
    return None

def show_game_statistics(games_data):
    """Show statistics about the created games."""
    console.print("\n[bold magenta]📊 Game Statistics[/bold magenta]")
    
    table = Table(title="TextWorld Games Overview")
    table.add_column("Game", style="cyan")
    table.add_column("Rooms", style="green")
    table.add_column("Objects", style="yellow")
    table.add_column("Connections", style="blue")
    
    for name, game in games_data:
        rooms = len(game.world.rooms)
        objects = len([obj for obj in game.world.objects if obj.type != 'P'])  # Exclude player
        connections = sum(len(room.exits) for room in game.world.rooms)
        
        table.add_row(name, str(rooms), str(objects), str(connections))
    
    console.print(table)

def demonstrate_textworld_features():
    """Demonstrate various TextWorld features for RL development."""
    console.print(Panel(
        "[bold green]🌟 TextWorld Features for RL Development[/bold green]",
        title="Features Available"
    ))
    
    features = [
        ("✅ Game Creation", "Build custom text-based environments"),
        ("✅ Room Navigation", "Multi-room environments with connections"),
        ("✅ Object Interaction", "Pick up, drop, and manipulate objects"),
        ("✅ Inventory Management", "Track player inventory state"),
        ("✅ Game State Tracking", "Monitor score, completion status"),
        ("✅ Action Processing", "Process natural language commands"),
        ("✅ Feedback Generation", "Get descriptive text observations"),
        ("❌ PDDL Planning", "Complex automated planning (not available)"),
        ("❌ Advanced Logic", "Complex rule systems (limited without PDDL)")
    ]
    
    for feature, description in features:
        console.print(f"{feature} {description}")

def main():
    """Main demonstration function."""
    console.print(Panel(
        "[bold blue]🎮 TextWorld Interactive Demo[/bold blue]\n"
        "Demonstrating TextWorld environments without PDDL dependencies",
        title="TextWorld Demo"
    ))
    
    # Create different types of games
    treasure_game = create_treasure_hunt_game()
    quest_game = create_simple_quest_game()
    
    games_data = [
        ("Treasure Hunt", treasure_game),
        ("Simple Quest", quest_game)
    ]
    
    # Show game statistics
    show_game_statistics(games_data)
    
    # Demonstrate interaction with each game
    for game_name, game in games_data:
        game_process = demonstrate_game_interaction(game, game_name)
    
    # Show available features
    demonstrate_textworld_features()
    
    # Usage recommendations
    console.print(Panel(
        "[bold green]💡 Usage Recommendations[/bold green]\n\n"
        "[yellow]For RL Training:[/yellow]\n"
        "• Use TextWorld games as environments for language agents\n"
        "• Train agents to navigate and complete objectives\n"
        "• Implement reward functions based on game score/completion\n\n"
        "[yellow]For Agent Development:[/yellow]\n"
        "• Test natural language understanding capabilities\n"
        "• Develop action planning without complex PDDL\n"
        "• Create custom environments for specific tasks\n\n"
        "[yellow]Limitations:[/yellow]\n"
        "• No automated planning (PDDL features unavailable)\n"
        "• Manual game design required for complex scenarios\n"
        "• Limited to basic text-based interactions",
        title="🚀 Next Steps"
    ))

if __name__ == "__main__":
    main()
