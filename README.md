# OXCEGameBot
Automated Controller for OpenXcom Extended (WiP)

## 🎮 Features
✅ **Working:**  
- Game process management
- Save game parsing (YAML)
- Screenshot capture via OpenCV
- Keyboard/mouse input control
- Basic state machine

🚀 **In Progress:**  
- Monte Carlo Tree Search decision making
- Computer vision for UI element detection
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
    - Open ```OXCEBot.py``` in your favorite editor.
    - Change ```GAME_BINARY_PATH``` and ```SAVE_DIRECTORY_PATH``` to the actual paths to your game.
    - Save.

7. Run
```bash
python3.11 ./OXCEBot.py
```
