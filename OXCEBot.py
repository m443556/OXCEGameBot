#!/usr/bin/env python3.11
"""
OXCE Game Bot - Automated Controller for OpenXcom Extended
Version 0.008 (MCTS)
"""

import subprocess
import time
import os
import signal
from pynput import mouse, keyboard
from pynput.mouse import Button
from pynput.keyboard import Key, Listener
import psutil
import cv2
import yaml
import logging
from pathlib import Path
import copy
import random
import math

# Configure logging with consistent format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("OXCEBot")

class MCTSNode:
    """Node for Monte Carlo Tree Search."""

    def __init__(self, state, parent=None, action=None):
        """
        Initialize an MCTS node.

        Args:
            state: The game state (parsed YAML data)
            parent: Parent node
            action: Action that led to this state
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.wins = 0.0  # Use float for fractional rewards
        self.untried_actions = self.get_legal_actions()

    def get_legal_actions(self):
        """Get list of legal actions from current state."""
        # TODO: Implement game-specific logic to determine possible actions based on self.state
        # For example, in battlescape: move unit, shoot, end turn, etc.
        # Actions could be represented as dicts or strings, e.g., {'type': 'move', 'unit_id': 1, 'position': (x, y)}
        return []

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        """Select the best child node using UCB1 formula."""
        choices_weights = [
            (child.wins / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def is_terminal(self):
        """Check if the state is a terminal state (game over)."""
        # TODO: Implement game-specific logic to check if the game is won/lost
        return False

class OXCEGameBot:
    """Main controller class for automating OpenXcom Extended gameplay."""

    # Constants for key delays and timeouts
    KEY_PRESS_DELAY = 0.05
    GAME_START_DELAY = 2.0
    SAVE_LOAD_DELAY = 1.0
    SCREENSHOT_DELAY = 1.0
    PROCESS_TERMINATION_TIMEOUT = 5.0

    # Default File Names
    FILE_NAME_AUTOSAVE = "_quick_.asav"
    FILE_NAME_SCREENSHOT = "screen000.png"

    def __init__(self, game_binary_path: str, save_directory_path: str):
        """
        Initialize the game bot controller.

        Args:
            game_binary_path: Path to the OXCE executable binary
            save_directory_path: Path to the game save directory
        """
        self.game_binary_path = Path(game_binary_path)
        self.save_directory_path = Path(save_directory_path)
        self.game_window_title = "OpenXcom"  # Default window title

        # Process management
        self.game_process = None
        self.game_process_id = None
        self.is_game_running = False
        self.should_continue_running = True
        self.is_bot_processing_enabled = False

        # Input controllers
        self.mouse_controller = mouse.Controller()
        self.keyboard_controller = keyboard.Controller()

        # Input event listeners
        self.keyboard_listener = Listener(on_press=self.on_key_press)

        logger.info("OXCE Game Bot initialized successfully")

    def launch_game(self) -> bool:
        """
        Start the OpenXcom Extended game process.

        Returns:
            bool: True if game started successfully, False otherwise
        """
        try:
            # Validate game binary exists
            if not self.game_binary_path.exists():
                logger.error(f"Game binary not found at: {self.game_binary_path.absolute()}")
                return False

            # Start the game process
            self.game_process = subprocess.Popen(
                [str(self.game_binary_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.game_process_id = self.game_process.pid

            logger.info(f"Game process started with PID: {self.game_process_id}")

            # Initialize keyboard listener before game takes focus
            self.keyboard_listener.start()
            logger.info("Keyboard listener activated successfully")

            # Allow time for game initialization
            time.sleep(self.GAME_START_DELAY)

            # Verify game is still running
            if self.game_process.poll() is None:
                self.is_game_running = True
                logger.info("OpenXcom Extended launched successfully")
                return True
            else:
                logger.error("Game process terminated during startup")
                return False

        except Exception as error:
            logger.error(f"Failed to launch game: {str(error)}", exc_info=True)
            return False

    def is_game_process_active(self) -> bool:
        """
        Check if the game process is still running.

        Returns:
            bool: True if game is running, False otherwise
        """
        if not self.game_process or not self.is_game_running:
            return False

        # Check process status
        if self.game_process.poll() is not None:
            logger.info("Game process has terminated normally")
            self.is_game_running = False
            return False

        # Additional verification using psutil
        try:
            if not psutil.pid_exists(self.game_process_id):
                logger.info("Game process no longer exists (psutil verification)")
                self.is_game_running = False
                return False
        except Exception as error:
            logger.debug(f"Error verifying process status: {str(error)}")

        return True

    def terminate_game(self) -> None:
        """Gracefully terminate the game process and clean up resources."""
        if self.game_process and self.is_game_running:
            try:
                # Attempt graceful termination first
                if psutil.pid_exists(self.game_process_id):
                    parent_process = psutil.Process(self.game_process_id)

                    # Terminate child processes recursively
                    child_processes = parent_process.children(recursive=True)
                    for child in child_processes:
                        logger.debug(f"Terminating child process: {child.pid}")
                        child.terminate()

                    # Terminate main process
                    logger.info("Initiating graceful game termination")
                    parent_process.terminate()
                    parent_process.wait(timeout=self.PROCESS_TERMINATION_TIMEOUT)

                self.is_game_running = False
                logger.info("Game terminated successfully")

            except (psutil.NoSuchProcess, psutil.TimeoutExpired, psutil.AccessDenied) as error:
                logger.warning(f"Graceful termination failed: {str(error)}. Forcing termination.")
                try:
                    self.game_process.kill()
                    self.is_game_running = False
                    logger.info("Game process force-terminated")
                except Exception as force_error:
                    logger.error(f"Force termination failed: {str(force_error)}")
            except Exception as error:
                logger.error(f"Error during game termination: {str(error)}", exc_info=True)

        # Clean up keyboard listener
        if hasattr(self, 'keyboard_listener') and self.keyboard_listener.is_alive():
            try:
                logger.info("Stopping keyboard listener")
                self.keyboard_listener.stop()
                self.keyboard_listener.join(timeout=1.0)
                logger.info("Keyboard listener stopped successfully")
            except Exception as error:
                logger.error(f"Error stopping keyboard listener: {str(error)}", exc_info=True)

    def main_loop(self) -> None:
        """Main bot execution loop that processes game state and makes decisions."""
        if not self.is_game_running:
            logger.error("Bot cannot start - game is not running. Call launch_game() first.")
            return

        try:
            logger.info("=" * 50)
            logger.info("Bot main loop started")
            logger.info("Available controls:")
            logger.info("  'l' - Toggle bot processing (on/off)")
            logger.info("  'q' - Quit bot and terminate game")
            logger.info("=" * 50)

            while self.should_continue_running and self.is_game_running:
                # Verify game is still active
                if not self.is_game_process_active():
                    logger.info("Game has exited - stopping bot operations")
                    self.should_continue_running = False
                    break

                # Process game state if bot is enabled
                if self.is_bot_processing_enabled:
                    self.process_game_state()

                # Check for termination signal
                if not self.should_continue_running:
                    break

                # Maintain reasonable loop frequency
                time.sleep(0.5)

        except KeyboardInterrupt:
            logger.info("Bot execution interrupted by user (Ctrl+C)")
            self.should_continue_running = False
        except Exception as error:
            logger.error(f"Critical error in main loop: {str(error)}", exc_info=True)
            self.should_continue_running = False
        finally:
            logger.info("Main loop execution completed")

    def toggle_bot_processing(self) -> None:
        """Toggle the bot's processing state between active and paused."""
        self.is_bot_processing_enabled = not self.is_bot_processing_enabled
        status = "ENABLED" if self.is_bot_processing_enabled else "DISABLED"
        logger.info(f"Bot processing state: {status}")

    def on_key_press(self, key) -> bool:
        """
        Handle keyboard input events for bot control.

        Args:
            key: The key pressed by the user

        Returns:
            bool: True to continue listening, False to stop
        """
        try:
            # Toggle bot processing with 'l' key
            if hasattr(key, 'char') and key.char == 'l':
                self.toggle_bot_processing()
                return True

            # Quit bot with 'q' key
            if hasattr(key, 'char') and key.char == 'q':
                logger.info("Quit command received - initiating shutdown sequence")
                self.should_continue_running = False
                return True

            return True  # Continue listening for other keys

        except Exception as error:
            logger.error(f"Error processing keyboard input: {str(error)}", exc_info=True)
            return True

    def process_game_state(self) -> None:
        """Capture and analyze current game state, then execute bot actions."""
        if not self.is_bot_processing_enabled:
            return

        try:
            logger.debug("Capturing game state...")

            # Capture game state data
            game_state_data = self.capture_savegame_state(self.FILE_NAME_AUTOSAVE)
            if game_state_data is None:
                logger.warning("Failed to capture game state - skipping processing cycle")
                return

            # Capture game screenshot
            screenshot_data = self.capture_game_screenshot(self.FILE_NAME_SCREENSHOT)
            if screenshot_data is None:
                logger.warning("Failed to capture screenshot - skipping processing cycle")
                return

            # Make decisions based on captured data
            bot_actions = self.make_decision(game_state_data, screenshot_data)

            # Execute actions (placeholder for future implementation)
            if bot_actions:
                logger.debug(f"Decision made: {bot_actions}")
                # self.execute_actions(bot_actions)

            logger.debug("Game state processing cycle completed successfully")

        except Exception as error:
            logger.error(f"Error processing game state: {str(error)}", exc_info=True)

    def capture_savegame_state(self, filename: str):
        """
        Trigger and load a quick savegame file containing current game state.

        Args:
            filename: Name of the savegame file to capture

        Returns:
            Game state data or None if failed
        """
        try:
            logger.debug(f"Triggering savegame capture: {filename}")

            # Trigger game to create save file
            self.keyboard_controller.press(Key.f5)
            self.keyboard_controller.release(Key.f5)
            time.sleep(self.SAVE_LOAD_DELAY)

            # Load the save file
            save_file_path = self.save_directory_path / filename
            return self.load_savegame_file(save_file_path)

        except Exception as error:
            logger.error(f"Error capturing savegame state: {str(error)}", exc_info=True)
            return None

    def load_savegame_file(self, file_path: Path):
        """
        Load and parse a savegame file from disk.

        Args:
            file_path: Path to the savegame file

        Returns:
            Parsed game state data or None if failed
        """
        try:
            if not file_path.exists():
                logger.error(f"Savegame file not found: {file_path.absolute()}")
                return None

            with open(file_path, 'r') as save_file:
                logger.debug(f"Loading savegame data from: {file_path}")
                return list(yaml.load_all(save_file, Loader=yaml.SafeLoader))

        except FileNotFoundError:
            logger.error(f"Savegame file not found: {file_path.absolute()}")
        except yaml.YAMLError as error:
            logger.error(f"YAML parsing error in savegame file: {str(error)}")
        except Exception as error:
            logger.error(f"Error loading savegame file: {str(error)}", exc_info=True)

        return None

    def capture_game_screenshot(self, filename: str):
        """
        Trigger and load a screenshot of the current game screen.

        Args:
            filename: Name of the screenshot file to capture

        Returns:
            Screenshot image data or None if failed
        """
        try:
            screenshot_path = self.save_directory_path / filename

            # Clean up any existing screenshot file
            if screenshot_path.exists():
                logger.debug(f"Removing existing screenshot: {screenshot_path}")
                screenshot_path.unlink()
                time.sleep(self.KEY_PRESS_DELAY)

            # Trigger game to create new screenshot
            logger.debug("Triggering screenshot capture")
            self.keyboard_controller.press(Key.f12)
            self.keyboard_controller.release(Key.f12)
            time.sleep(self.SCREENSHOT_DELAY)

            # Load the screenshot
            screenshot_data = self.load_image_file(screenshot_path)

            # Clean up the screenshot file
            if screenshot_path.exists():
                screenshot_path.unlink()
                logger.debug(f"Cleaned up screenshot file: {screenshot_path}")

            return screenshot_data

        except Exception as error:
            logger.error(f"Error capturing screenshot: {str(error)}", exc_info=True)
            return None

    def load_image_file(self, file_path: Path):
        """
        Load an image file from disk using OpenCV.

        Args:
            file_path: Path to the image file

        Returns:
            Image data as numpy array or None if failed
        """
        try:
            if not file_path.exists():
                logger.error(f"Screenshot file not found: {file_path.absolute()}")
                return None

            logger.debug(f"Loading image from: {file_path}")
            image = cv2.imread(str(file_path))

            if image is not None:
                logger.debug(f"Image loaded successfully - dimensions: {image.shape}")
                return image
            else:
                logger.error(f"Failed to load image data from: {file_path.absolute()}")

        except Exception as error:
            logger.error(f"Error loading image file: {str(error)}", exc_info=True)

        return None

    def apply_action(self, state, action):
        """
        Apply an action to the state and return the new state.

        Args:
            state: Current game state
            action: Action to apply

        Returns:
            New state after applying action
        """
        # TODO: Implement game-specific state transition logic
        # This requires modeling the game rules in Python, e.g., update unit positions, resolve combats, etc.
        # For complex games like OpenXcom, this might involve simplifying assumptions or partial modeling.
        # Note: Use copy.deepcopy(state) as base to avoid modifying original.
        new_state = copy.deepcopy(state)
        return new_state

    def simulate(self, state):
        """
        Perform a random simulation (rollout) from the given state to a terminal state.

        Args:
            state: Starting state for simulation

        Returns:
            Reward (e.g., 1 for win, 0 for loss, or a score)
        """
        # TODO: Implement game-specific simulation
        # Repeatedly apply random actions until terminal state, then evaluate.
        current_state = copy.deepcopy(state)
        while not MCTSNode(current_state).is_terminal():
            legal_actions = MCTSNode(current_state).get_legal_actions()
            if not legal_actions:
                break
            action = random.choice(legal_actions)
            current_state = self.apply_action(current_state, action)
        # TODO: Evaluate the terminal state (e.g., based on victory conditions in YAML)
        return random.random()  # Placeholder: random reward between 0 and 1

    def make_decision(self, game_state_data, screenshot_data):
        """
        Analyze game state and make strategic decisions using Monte Carlo Tree Search.

        Args:
            game_state_data: Parsed game state from save file
            screenshot_data: Current game screen image

        Returns:
            Decision object containing actions to execute
        """
        logger.debug("Starting MCTS decision making")

        # Process game_state_data (assuming first document is the main state)
        if isinstance(game_state_data, list) and game_state_data:
            state = copy.deepcopy(game_state_data[0])
        else:
            state = copy.deepcopy(game_state_data) if game_state_data else {}

        # Optional: Use screenshot_data for additional analysis (e.g., CV to detect UI elements)
        # TODO: Integrate CV if needed, e.g., detect game phase if not in YAML

        root = MCTSNode(state)

        # Number of MCTS iterations (adjust based on time constraints)
        num_iterations = 100  # Small number for placeholder; increase for better decisions

        for _ in range(num_iterations):
            node = root

            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child()

            # Expansion
            if not node.is_fully_expanded():
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                new_state = self.apply_action(node.state, action)
                child = MCTSNode(new_state, parent=node, action=action)
                node.children.append(child)
                node = child

            # Simulation
            simulation_result = self.simulate(node.state)

            # Backpropagation
            while node is not None:
                node.visits += 1
                node.wins += simulation_result
                node = node.parent

        # Select best action (child with highest visit count)
        if not root.children:
            logger.warning("No actions expanded in MCTS")
            return {"action": "wait", "confidence": 0.0}

        best_child = max(root.children, key=lambda c: c.visits)
        confidence = best_child.wins / best_child.visits if best_child.visits > 0 else 0.0
        logger.debug(f"Best action selected with confidence: {confidence}")

        return {"action": best_child.action, "confidence": confidence}

# ======================
# Entry Point
# ======================

def main() -> None:
    """
    Main entry point for the OXCE Game Bot application.
    Initializes the bot, launches the game, and starts the main loop.
    """
    # Configuration - Update these paths for your system
    GAME_BINARY_PATH = "/home/user/openxcomex/OpenXcomEx"
    SAVE_DIRECTORY_PATH = "/home/user/.local/share/openxcom/40k"

    # Initialize bot controller
    bot_controller = OXCEGameBot(GAME_BINARY_PATH, SAVE_DIRECTORY_PATH)

    try:
        logger.info("Starting OXCE Game Bot...")

        # Launch the game
        if not bot_controller.launch_game():
            logger.critical("Game launch failed - aborting bot execution")
            return

        logger.info("Bot operations commenced")
        bot_controller.main_loop()
        logger.info("Bot operations completed successfully")

    except KeyboardInterrupt:
        logger.info("Bot execution interrupted by user (Ctrl+C)")
    except Exception as error:
        logger.critical(f"Critical failure in main execution: {str(error)}", exc_info=True)
    finally:
        logger.info("Initiating clean shutdown sequence")
        bot_controller.terminate_game()
        logger.info("OXCE Game Bot shutdown complete - exiting")

if __name__ == "__main__":
    main()
