# -*- coding: utf-8 -*-
# Required for explicit UTF-8 encoding if needed, especially for comments/strings

import sys
import os
import pickle
import numpy as np

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.properties import ObjectProperty
from kivy.core.text import LabelBase # <-- Import LabelBase for font registration
from kivy.graphics import Color, Rectangle # <-- Import for background
# from kivy.uix.widget import Widget # Import if needed for Spacer
from kivy.clock import Clock # Import Clock for scheduling AI move

from common import GameBoard
from train import NeuralNetwork  # 导入 NeuralNetwork 类

# --- Helper Function for Resource Paths (Dev vs. PyInstaller) ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        # Not running in a PyInstaller bundle
        base_path = os.path.abspath(os.path.dirname(__file__))

    return os.path.join(base_path, relative_path)

# --- Define Resource Paths ---
MODEL_PATH = resource_path('tic_tac_toe_selfplay_nn_v1.pkl') # Or 'best_model.pkl'
FONT_PATH = resource_path('assets/fonts/SourceHanSansSC-Normal-2.otf')
FONT_NAME = 'SourceHanSans' # Choose a name for registration

# --- Register Custom Font ---
try:
    LabelBase.register(name=FONT_NAME, fn_regular=FONT_PATH)
except Exception as e:
     print(f"Warning: Could not register font {FONT_NAME} from {FONT_PATH}. Error: {e}")
     FONT_NAME = 'Roboto' # Fallback

class AIManager:
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)

    def load_model(self, model_path: str):
        try:
            print(f"Attempting to load model from: {model_path}")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                print(f"Successfully loaded model object of type: {type(model)}")
                if not hasattr(model, 'get_best_move'):
                    print(f"Error: Loaded object does not have 'get_best_move' method.")
                    raise TypeError("Loaded model object is missing 'get_best_move'.")
                return model
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def predict_move(self, board: list[list[str | None]]) -> int | None:
        if not self.model:
            print("Error: AI model not loaded.")
            return None
        try:
            inputs = self.board_to_input(board)
            move_index = self.model.get_best_move(inputs, board, strategy='greedy')
            return move_index
        except Exception as e:
            print(f"Error during AI prediction: {e}")
            return None

    def board_to_input(self, GameState: list[list[str | None]], col: int = 3, row: int = 3) -> np.ndarray:
        if not isinstance(GameState, list) or not all(isinstance(r, list) for r in GameState):
            print(f"Warning: Invalid GameState in board_to_input: {GameState}")
            return np.zeros((col * row * 2))

        board = np.zeros((col, row, 2))
        try:
            for i in range(col):
                for j in range(row):
                    if i < len(GameState) and j < len(GameState[i]):
                        cell = GameState[i][j]
                        if cell == 'X' or cell == 'x': # AI Player
                            board[i, j, 0] = 1
                        elif cell == 'O' or cell == 'o': # Human Player
                            board[i, j, 1] = 1
                    else:
                         print(f"Warning: Out of bounds index [{i}][{j}] in GameState")
        except IndexError as e:
            print(f"Error processing board state: {e}. GameState: {GameState}")
            return np.zeros((col * row * 2))
        return board.flatten()

class GameGrid(GridLayout):
    def __init__(self, **kwargs):
        super(GameGrid, self).__init__(**kwargs)
        self.cols = 3
        # self.rows = 3 # Implicitly 3x3 due to board_size
        self.padding = 5
        self.spacing = 5
        self.board = GameBoard()
        self.buttons = []
        self._ai_enabled = False
        self.game_over_flag = False # Flag to prevent moves after game ends

        # --- Initialize AI Manager ---
        try:
            self.ai_manager = AIManager(MODEL_PATH)
            self._ai_enabled = True
            print("AI Manager initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize AI Manager: {e}")
            self.ai_manager = None

        # --- Create Buttons ---
        for i in range(self.board.board_size):
            btn = Button(text='', font_size='40sp')
            btn.bind(on_press=self.on_button_press)
            self.add_widget(btn)
            self.buttons.append(btn)

        # --- Set Initial Game State and Trigger First AI Move ---
        # AI ('X') is player 0, Human ('O') is player 1
        self.board.current_player = 0 # *** AI ('X') starts ***
        self.game_over_flag = False

        # Schedule the first AI move slightly after the UI is built
        # This prevents potential issues with accessing widgets too early
        # Only schedule if AI is actually enabled
        if self._ai_enabled:
            Clock.schedule_once(self.trigger_initial_ai_move, 0.1) # Delay of 0.1 seconds
            # Note: Status label update happens in GameControls init
        else:
             # If AI failed, game cannot start properly with AI first
             self.disable_buttons() # Maybe disable board if AI needed but failed


    def trigger_initial_ai_move(self, dt):
        """ Wrapper to call ai_move safely after initialization. """
        print("Triggering initial AI move...")
        self.ai_move()
        # Update status *after* AI moves
        if not self.game_over_flag and self.parent and hasattr(self.parent, 'status_label'):
            self.parent.status_label.text = "轮到你了 ('O')" # Your turn ('O')


    def on_button_press(self, instance):
        if self.game_over_flag:
            print("Game is over. Ignoring click.")
            return

        if not self._ai_enabled:
             print("AI is not available.")
             # Maybe show error message? Status label should already reflect this.
             return

        # Check if it's Human's turn (player 1) before processing click
        if self.board.current_player == 1:
            index = self.buttons.index(instance)
            row, col = index // 3, index % 3

            if instance.text == '' and self.board.board[row][col] is None:
                print(f"Player 'O' clicked button {index} ([{row}][{col}])")
                instance.text = 'O'
                self.board.board[row][col] = 'O'

                if not self.check_game_status(): # Check if game ended
                    self.board.current_player = 0 # Switch to AI's turn
                    self.update_status("AI ('X') 正在思考...") # AI ('X') is thinking...
                    # Schedule AI move to allow UI to update status first
                    Clock.schedule_once(lambda dt: self.ai_move(), 0.1)
                # else: game_over handled within check_game_status

            elif instance.text != '':
                print(f"Button {index} already taken ({instance.text}).")
            elif self.board.board[row][col] is not None:
                 print(f"Board/Button mismatch at {index}. Board: {self.board.board[row][col]}, Button: {instance.text}")
                 instance.text = self.board.board[row][col] # Sync button
            # No 'else' needed as the outer 'if' checks for player turn

        elif self.board.current_player == 0:
             print("Not your turn (AI is thinking).")
        else:
            # Should not happen
            print(f"Unexpected player index: {self.board.current_player}")

    def ai_move(self):
        print("Attempting AI move...")
        if self.game_over_flag:
             print("AI move skipped: Game already over.")
             return

        # AI moves only if enabled AND it's its turn (player 0)
        if self._ai_enabled and self.ai_manager and self.board.current_player == 0:
            move_index = self.ai_manager.predict_move(self.board.board)
            print(f"AI predicted move index: {move_index}")

            valid_move_made = False
            if move_index is not None and 0 <= move_index < len(self.buttons):
                # Check if the chosen cell is actually empty
                if self.buttons[move_index].text == '':
                    row, col = move_index // 3, move_index % 3
                    if self.board.board[row][col] is None:
                         print(f"AI placing 'X' at index {move_index} ([{row}][{col}])")
                         self.buttons[move_index].text = 'X'
                         self.board.board[row][col] = 'X'
                         valid_move_made = True
                         if not self.check_game_status(): # Check if game ended
                            self.board.current_player = 1 # Switch back to Player's turn
                            self.update_status("轮到你了 ('O')") # Your turn ('O')
                         # else: game_over handled within check_game_status
                    else:
                        print(f"AI Error: Predicted move {move_index} but board[{row}][{col}] is not None ({self.board.board[row][col]}).")
                        # *** Potential infinite loop if AI keeps picking filled spot ***
                        # Need a fallback here - maybe pick random valid move?
                        self.handle_ai_prediction_error()
                else:
                     print(f"AI Error: Predicted move index {move_index} but button text is not empty ('{self.buttons[move_index].text}').")
                     self.handle_ai_prediction_error()

            elif move_index is None:
                 print("AI prediction failed (returned None).")
                 self.handle_ai_prediction_error() # Treat as error
            else:
                 print(f"AI Error: Predicted invalid move index: {move_index}")
                 self.handle_ai_prediction_error() # Treat as error

            # If AI failed to make a valid move, potentially lock up game.
            # For now, handle_ai_prediction_error might just print. A better
            # implementation would try a different move.

        elif not self._ai_enabled:
             print("AI move skipped: AI is disabled.")
             # Game shouldn't reach here if AI starts first and is disabled (buttons disabled)
        elif self.board.current_player != 0:
             print(f"AI move skipped: Not AI's turn (Player is {self.board.current_player}).")
        # else: AI manager is None (already checked by _ai_enabled basically)

    def handle_ai_prediction_error(self):
        """ Basic handler for when AI fails to make a valid move. """
        print("AI failed to make a valid move. Game might be stuck.")
        # Option 1: Declare draw/loss for AI?
        # Option 2: Try a random valid move?
        available_moves = []
        for i in range(self.board.board_size):
            row, col = i // 3, i % 3
            if self.board.board[row][col] is None:
                available_moves.append(i)

        if available_moves:
            print("Attempting random fallback move...")
            random_index = np.random.choice(available_moves)
            row, col = random_index // 3, random_index % 3
            print(f"AI placing 'X' randomly at index {random_index} ([{row}][{col}])")
            self.buttons[random_index].text = 'X'
            self.board.board[row][col] = 'X'
            if not self.check_game_status():
                self.board.current_player = 1
                self.update_status("轮到你了 ('O')")
        else:
            # No moves left, should have been caught by draw check?
            print("AI prediction error, but no available moves found. Checking status again.")
            self.check_game_status() # Re-check for draw/win


    def check_game_status(self):
        """Checks for win/loss/draw and updates status/flags."""
        if self.game_over_flag: # Don't re-check if already over
            return True

        print("Checking game status...")
        board_state = self.board.board
        winner = None

        # Check rows
        for r in range(3):
            if board_state[r][0] is not None and board_state[r][0] == board_state[r][1] == board_state[r][2]:
                winner = board_state[r][0]
                break
        # Check columns
        if winner is None:
            for c in range(3):
                if board_state[0][c] is not None and board_state[0][c] == board_state[1][c] == board_state[2][c]:
                    winner = board_state[0][c]
                    break
        # Check diagonals
        if winner is None:
            center_cell = board_state[1][1]
            if center_cell is not None:
                if board_state[0][0] == center_cell == board_state[2][2]:
                    winner = center_cell
                elif board_state[0][2] == center_cell == board_state[2][0]:
                    winner = center_cell

        # If winner found
        if winner:
            print(f"Winner found: {winner}")
            self.game_over(winner)
            return True

        # Check for draw (no winner and board is full)
        is_full = all(cell is not None for row in board_state for cell in row)
        if is_full:
            print("Board is full, game is a draw.")
            self.game_over('Draw')
            return True

        print("Game continues.")
        return False # Game continues

    def game_over(self, result):
        """ Handles the end of the game. """
        print(f"Game Over! Result: {result}")
        self.game_over_flag = True # Set the flag
        self.disable_buttons()

        status_text = ""
        if result == 'X':
            status_text = "人类你输了 (AI Wins 'X')"
        elif result == 'O':
            status_text = "人类你赢了 (Player Wins 'O')"
        elif result == 'Draw':
            status_text = "平局 (Draw)"
        else:
            status_text = f"游戏结束 - 未知: {result}"

        self.update_status(status_text)


    def disable_buttons(self):
        """ Disables all grid buttons. """
        print("Disabling game buttons.")
        for btn in self.buttons:
            btn.disabled = True

    def reset_game(self):
        """ Resets the game board and UI elements. """
        print("Resetting game...")
        self.board.reset_board()
        self.game_over_flag = False # Reset game over flag

        for i, btn in enumerate(self.buttons):
            btn.text = ''
            btn.disabled = False # Re-enable buttons
            # Ensure internal board is also clear (reset_board should do this)
            row, col = i // 3, i % 3
            if self.board.board[row][col] is not None:
                 print(f"Warning: Board cell [{row}][{col}] not None after reset. Forcing.")
                 self.board.board[row][col] = None

        # --- Set AI as starting player and trigger its move ---
        self.board.current_player = 0 # *** AI ('X') starts ***

        # Update status and trigger AI move, only if AI is enabled
        if self._ai_enabled:
            self.update_status("AI ('X') 正在思考...") # AI ('X') is thinking...
            # Use Clock schedule again for safety/consistency
            Clock.schedule_once(self.trigger_initial_ai_move, 0.1)
        else:
            # AI failed to load, show error and disable board
            self.update_status("错误：AI未加载！") # Error: AI not loaded!
            self.disable_buttons()

    def update_status(self, message: str):
        """ Safely updates the status label text via the parent. """
        if self.parent and hasattr(self.parent, 'status_label'):
            self.parent.status_label.text = message
        else:
             print(f"Status Update (Label not found): {message}")


class GameControls(BoxLayout):
    def __init__(self, **kwargs):
        super(GameControls, self).__init__(**kwargs)
        self.orientation = 'horizontal'
        self.padding = 10
        self.spacing = 10

        with self.canvas.before:
            Color(0.95, 0.95, 0.95, 1) # Slightly lighter grey
            self._background_rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_background, pos=self._update_background)

        # --- Left Side: Game Grid ---
        self.game_grid = GameGrid(size_hint=(0.7, 1))
        self.add_widget(self.game_grid)

        # --- Right Side: Control Panel ---
        control_panel = BoxLayout(
            orientation='vertical',
            size_hint=(0.3, 1),
            spacing=10,
            padding=10
        )

        # Title Label
        control_panel.add_widget(Label(
            text='[b]井字棋 AI 对战[/b]',
            markup=True,
            font_size='24sp',
            font_name=FONT_NAME,
            color=[0.1, 0.1, 0.1, 1],
            size_hint_y=None,
            height=50
        ))

        # Status Label (Initialized based on AI status AFTER grid init)
        self.status_label = Label(
            # Initial text set below
            font_size='18sp',
            font_name=FONT_NAME,
            color=[0.2, 0.2, 0.2, 1],
            size_hint=(1, 0.3), # Take horizontal space, fixed vertical portion
            # halign='center', # Alignment might need text_size binding
            # valign='top'
        )
        # Bind size to update text_size for alignment/wrapping
        def update_text_size(instance, size):
            instance.text_size = (size[0], None) # Width constraint, auto height
        self.status_label.bind(size=update_text_size)
        control_panel.add_widget(self.status_label)


        # --- Spacer ---
        # control_panel.add_widget(Widget(size_hint_y=0.4)) # Pushes button down

        # --- Reset Button ---
        reset_button = Button(
            text="重置游戏",
            font_size='16sp',
            font_name=FONT_NAME,
            size_hint_y=None, # Fixed height
            height=50,
            background_color=[0.2, 0.6, 0.8, 1],
            color=[1, 1, 1, 1]
            # background_normal='', # Needed if using background_color effectively
        )
        reset_button.bind(on_press=self.reset_game_action)
        control_panel.add_widget(reset_button)

        self.add_widget(control_panel)

        # --- Set Initial Status Label Text ---
        # This runs AFTER self.game_grid has been initialized and attempted AI load
        if self.game_grid._ai_enabled:
             # Initial status shown before AI makes its very first move (via Clock)
            self.status_label.text = "游戏开始\nAI ('X') 正在思考..." # Game start, AI thinking..
        else:
            self.status_label.text = "错误：无法加载AI模型！\n请检查模型文件。"
            self.status_label.color = [0.8, 0.1, 0.1, 1]

    def _update_background(self, instance, value):
        self._background_rect.pos = instance.pos
        self._background_rect.size = instance.size

    def reset_game_action(self, instance):
        print("Reset button pressed (GameControls)")
        # Delegate reset to the grid, which handles AI first move etc.
        self.game_grid.reset_game()
        # Status label is updated within game_grid.reset_game or subsequent ai_move


class TicTacToeApp(App):
    def build(self):
        self.title = '井字棋 AI 对战 (Tic Tac Toe AI) - AI First'
        # from kivy.core.window import Window
        # Window.size = (600, 400)
        return GameControls()

if __name__ == '__main__':
    TicTacToeApp().run()
