# OXCEGameBot
Automated Controller for OpenXcom Extended (![Project Status](https://img.shields.io/badge/status-WIP-yellow))

## Overview
A Python-based bot that automates gameplay for OpenXcom Extended (OXCE) using computer vision and game state analysis.

## Features
### ✅ What Works:
- Launching OXCE from python wrapper
- Parsing all data from quicksave files
- Taking screenshots for OpenCV analysis
- Sending keyboard and mouse events to OXCE

### 🚧 What's In Progress:
- Advanced reasoning via Monte Carlo Tree Search (MCTS)
- Computer vision for UI element detection and processing
- Executing selected in-game actions autonomously

### 🚀 Future Plans:
- Tactical pathfinding algorithms
- Multi-unit coordination

## Quick Start
1. Get the latest OXCE for Linux
   
3. Install Python (recommended 3.11)
   
5. Get bot files
```bash
git clone https://github.com/m443556/OXCEGameBot.git
```

4. Prepare the environment
```bash
cd OXCEGameBot
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

5. Set configuration variables!
    - Open ```OXCEBot.py``` in your favorite editor
    - Change ```GAME_BINARY_PATH``` and ```SAVE_DIRECTORY_PATH``` to the actual paths to your game
    - Save

7. Run
```bash
python3.11 ./OXCEBot.py
```

## Contributing
We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to get involved.

## License
This project is licensed under the BSD 3-Clause license - see the [LICENSE](LICENSE) file for details.
