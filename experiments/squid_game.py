import random


def simulate_game(disadvantage: int = 0):
    if disadvantage not in [0, 1, 2, 3]:
        raise ValueError("disadvantage must be either 0, 1, 2 or 3")

    # Define the buttons
    buttons = ["red", "grey", "green"]
    # Shuffle the buttons
    random.shuffle(buttons)

    # Player 1 presses a button
    button1 = buttons.pop(0)

    # Player 2 presses a button
    button2 = buttons.pop(0)

    # Player 3 presses the last button
    button3 = buttons[0]

    # If the player who gets the green button chose the other winner randomly.
    if disadvantage == 0:
        if button1 == "green":
            return ["player1", random.choice(["player2", "player3"])]
        elif button1 == "red":
            return ["player2", "player3"]
        # button1 == 'grey'
        elif button2 == "green":
            return ["player2", random.choice(["player1", "player3"])]
        elif button2 == "red":
            return ["player1", "player3"]
    # If the player1 is disliked by other players.
    if disadvantage == 1:
        if button1 == "green":
            return ["player1", random.choice(["player2", "player3"])]
        elif button1 == "red":
            return ["player2", "player3"]
        # button1 == 'grey'
        elif button2 == "green":
            return ["player2", "player3"]
        elif button2 == "red":
            return ["player1", "player3"]
    # If the player2 is disliked by other players.
    if disadvantage == 2:
        if button1 == "green":
            return ["player1", "player3"]
        elif button1 == "red":
            return ["player2", "player3"]
        # button1 == 'grey'
        elif button2 == "green":
            return ["player2", random.choice(["player1", "player3"])]
        elif button2 == "red":
            return ["player1", "player3"]
    # If the player3 is disliked by other players.
    if disadvantage == 3:
        if button1 == "green":
            return ["player1", "player2"]
        elif button1 == "red":
            return ["player2", "player3"]
        # button1 == 'grey'
        elif button2 == "green":
            return ["player2", "player1"]
        elif button2 == "red":
            return ["player1", "player3"]


def calculate_probabilities(n_simulations, disadvantage):
    # Initialize win counts
    win_counts = {"player1": 0, "player2": 0, "player3": 0}

    # Simulate the game n times
    for _ in range(n_simulations):
        winners = simulate_game(disadvantage)
        for winner in winners:
            win_counts[winner] += 1

    # Calculate probabilities
    probabilities = {
        player: win_count / n_simulations for player, win_count in win_counts.items()
    }

    return probabilities


for disadvantage in range(4):
    probabilities = calculate_probabilities(1000000, disadvantage)
    print(f"Disadvantage: Player{disadvantage}")
    for player, probability in probabilities.items():
        print(f"The probability of {player} winning is {probability:.4f}")
    print()
