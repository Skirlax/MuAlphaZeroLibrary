import os

if "root.root" in os.listdir():
    exit(0)
    
    
os.makedirs("AlphaZeroTicTacToe",exist_ok=True)
os.chdir("AlphaZeroTicTacToe")
os.makedirs("Logs/ProgramLogs")
os.makedirs("Checkpoints/NetVersions/Temp")
os.makedirs("Checkpoints/Traces")