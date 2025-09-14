#!/usr/bin/env python3
"""
Simple TextWorld Test - Core Functionality Only

This is a minimal test to verify TextWorld core functionality works
without any PDDL dependencies or complex features.
"""

from rich.console import Console
from rich.panel import Panel

console = Console()

def test_basic_import():
    """Test basic TextWorld import."""
    try:
        import textworld
        console.print("✅ TextWorld imported successfully")
        console.print(f"TextWorld version: {textworld.__version__}")
        return True
    except Exception as e:
        console.print(f"❌ Failed to import TextWorld: {e}")
        return False

def test_game_maker():
    """Test basic GameMaker functionality."""
    try:
        import textworld
        
        console.print("\n[bold blue]Testing GameMaker...[/bold blue]")
        
        # Create a simple game maker
        M = textworld.GameMaker()
        console.print("✅ GameMaker created")
        
        # Create a simple room
        room = M.new_room("Test Room")
        console.print(f"✅ Room created: {room.name}")
        
        # Create a simple object
        obj = M.new(type='o', name='test_object')
        console.print(f"✅ Object created: {obj.name}")
        
        # Add object to room
        room.add(obj)
        console.print("✅ Object added to room")
        
        # Set player location
        M.set_player(room)
        console.print("✅ Player location set")
        
        return True
        
    except Exception as e:
        console.print(f"❌ GameMaker test failed: {e}")
        import traceback
        console.print(traceback.format_exc())
        return False

def test_simple_game_creation():
    """Test creating a very simple game."""
    try:
        import textworld
        
        console.print("\n[bold blue]Testing Simple Game Creation...[/bold blue]")
        
        M = textworld.GameMaker()
        
        # Create minimal game structure
        room1 = M.new_room("Start Room")
        item = M.new(type='o', name='apple')
        room1.add(item)
        M.set_player(room1)
        
        # Try to build the game
        game = M.build()
        console.print("✅ Simple game built successfully")
        console.print(f"Game type: {type(game)}")
        
        # Check game properties
        if hasattr(game, 'world'):
            console.print(f"Game has world with {len(game.world.rooms)} rooms")
            console.print(f"Game has {len(game.world.objects)} objects")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Simple game creation failed: {e}")
        import traceback
        console.print(traceback.format_exc())
        return False

def main():
    """Run all basic tests."""
    console.print(Panel(
        "[bold green]Simple TextWorld Core Test[/bold green]\n"
        "Testing basic TextWorld functionality without PDDL or gym dependencies",
        title="🧪 TextWorld Test"
    ))
    
    tests = [
        ("Import Test", test_basic_import),
        ("GameMaker Test", test_game_maker), 
        ("Simple Game Creation", test_simple_game_creation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        console.print(f"\n[yellow]Running {test_name}...[/yellow]")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    console.print("\n" + "="*50)
    console.print("[bold]Test Results Summary:[/bold]")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        console.print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    console.print(f"\nTests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        console.print(Panel(
            "[bold green]🎉 All tests passed![/bold green]\n\n"
            "TextWorld core functionality is working correctly.\n"
            "You can now use basic TextWorld features for:\n"
            "• Creating simple text-based games\n"
            "• Building rooms and adding objects\n" 
            "• Basic game mechanics without PDDL planning\n\n"
            "[yellow]Note: Complex planning features requiring PDDL are not available.[/yellow]",
            title="Success"
        ))
    else:
        console.print(Panel(
            f"[bold red]Some tests failed ({passed}/{len(tests)} passed)[/bold red]\n"
            "Check the error messages above for troubleshooting.",
            title="Partial Success"
        ))

if __name__ == "__main__":
    main()
