import subprocess
import time

# The parameter space we want to sweep
THRESHOLDS_TO_TEST = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
GAMES_PER_THRESHOLD = 10 
OPPONENTS = ["expander", "hunter"]

def run_match(threshold, opponent):
    """
    Runs a headless match via subprocess and parses the winner.
    (Adjust the command array to match your local CLI structure)
    """
    cmd = [
        "python", "competition/matchup.py",
        "--agent1", "agents/temp_agent", # Your upgraded agent
        "--agent2", f"agents/{opponent}",
        "--agent1_kwargs", f"threshold={threshold}", # Inject the parameter
        "--headless" 
    ]
    
    try:
        # Run the match and capture the stdout
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout
        
        # Parse output for the winner (Adjust this based on what your matchup.py prints)
        if "Winner: Player 1" in output or "Winner: agent1" in output:
            return 1 # We won
        elif "Winner: Player 2" in output or "Winner: agent2" in output:
            return 0 # We lost
        else:
            return 0.5 # Draw
            
    except subprocess.TimeoutExpired:
        # If the game hits the 1200 turn limit and hangs
        return 0.5

def optimize():
    print(f"{'Threshold':<10} | {'Opponent':<10} | {'Win Rate':<10} | {'Time'}")
    print("-" * 50)
    
    best_threshold = None
    best_win_rate = -1

    for threshold in THRESHOLDS_TO_TEST:
        for opponent in OPPONENTS:
            wins = 0
            draws = 0
            start_time = time.time()
            
            for _ in range(GAMES_PER_THRESHOLD):
                result = run_match(threshold, opponent)
                if result == 1:
                    wins += 1
                elif result == 0.5:
                    draws += 1
                    
            elapsed = time.time() - start_time
            
            # Calculate win rate (draws count as half a win)
            win_rate = (wins + (draws * 0.5)) / GAMES_PER_THRESHOLD
            
            print(f"{threshold:<10} | {opponent:<10} | {win_rate:<10.0%} | {elapsed:.1f}s")
            
            # Track the best overall performing threshold
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_threshold = threshold

    print("-" * 50)
    print(f"Optimal Threshold Found: {best_threshold} (Max Win Rate: {best_win_rate:.0%})")

if __name__ == "__main__":
    optimize()