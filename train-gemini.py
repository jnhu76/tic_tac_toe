import numpy as np
from common import GameBoard, BOARD_SIZE, BOARD_COL, BOARD_ROW
import pickle
import os  # For checking if model file exists

# --- Constants ---
NN_INPUT_SIZE = 18  # 3x3 board, 2 channels (my pieces, opponent pieces)
NN_HIDDEN_SIZE = 64 # Reduced hidden size, 100-200 might be overkill
NN_OUTPUT_SIZE = 9  # 9 possible moves


# --- Modified Neural Network Class ---
class NeuralNetwork:
    def __init__(self, input_size: int = NN_INPUT_SIZE, hidden_size: int = NN_HIDDEN_SIZE, output_size: int = NN_OUTPUT_SIZE):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # He initialization (good practice for ReLU variants)
        self.weights_ih = np.random.randn(input_size, self.hidden_size) * np.sqrt(2. / input_size)
        self.weights_ho = np.random.randn(self.hidden_size, output_size) * np.sqrt(2. / self.hidden_size)
        self.biases_h = np.zeros(self.hidden_size)
        self.biases_o = np.zeros(output_size)

        # Store activations for backprop
        self.inputs = np.zeros(input_size)
        self.hidden_raw = np.zeros(self.hidden_size) # Store pre-activation hidden state
        self.hidden = np.zeros(self.hidden_size)
        self.outputs_raw = np.zeros(output_size) # Store pre-activation output logits
        self.outputs = np.zeros(output_size)

    def leaky_relu(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0, x, x * alpha)

    def leaky_relu_derivative(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        # Derivative should be applied to the *pre-activation* value (self.hidden_raw in forward pass)
        # The input 'x' here during backward pass will be the pre-activation value
        return np.where(x > 0, 1.0, alpha)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability before exponentiation
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        # Ensure sum is not zero before division
        sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
        return exp_x / np.where(sum_exp_x == 0, 1, sum_exp_x) # Avoid division by zero

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        # Hidden layer
        self.hidden_raw = np.dot(self.inputs, self.weights_ih) + self.biases_h
        self.hidden = self.leaky_relu(self.hidden_raw)
        # Output layer
        self.outputs_raw = np.dot(self.hidden, self.weights_ho) + self.biases_o
        self.outputs = self.softmax(self.outputs_raw)
        return self.outputs

    def backward(self, targets: np.ndarray, learning_rate: float):
        """
        反向传播更新权重和偏置 (Q-learning style target).
        `targets` should be the target Q-values for the output layer.
        """
        # Ensure targets is a flat numpy array
        targets = np.asarray(targets).flatten()
        if targets.shape != self.outputs.shape:
             raise ValueError(f"Shape mismatch: targets {targets.shape}, outputs {self.outputs.shape}")

        # Calculate output layer error (delta_output)
        # For softmax output with a target vector (like in classification or Q-value regression),
        # the gradient w.r.t the pre-softmax activations (outputs_raw) is simply (outputs - targets).
        # Note the sign compared to the original code. Use (outputs - targets) and *subtract* during update,
        # OR use (targets - outputs) and *add* during update. We'll use (targets - outputs) and add.
        output_errors_post_softmax = targets - self.outputs

        # --- Important Note on Softmax Gradient ---
        # While (outputs - targets) is the gradient for softmax WITH cross-entropy loss,
        # we are doing regression on Q-values. A simple MSE-like gradient on the output
        # might be more intuitive: `output_errors = (targets - self.outputs)`.
        # Let's stick to `targets - self.outputs` for consistency with prior code structure's "+" update.
        output_errors = targets - self.outputs # Error signal for output layer update

        # Calculate hidden layer error (delta_hidden)
        # Error propagated back from output layer
        hidden_errors = np.dot(output_errors, self.weights_ho.T)
        # Apply derivative of activation function (Leaky ReLU derivative applied to *pre-activation* `hidden_raw`)
        hidden_errors *= self.leaky_relu_derivative(self.hidden_raw) # Pass pre-activation values

        # --- Weight and Bias Updates (using vector operations) ---
        # To use np.outer, inputs and errors need to be 1D.
        # Ensure inputs and hidden activations are treated appropriately if batching were used.
        # For single instance backprop:
        inputs_flat = self.inputs.flatten()
        hidden_flat = self.hidden.flatten()
        output_errors_flat = output_errors.flatten()
        hidden_errors_flat = hidden_errors.flatten()

        # Update output layer weights and biases
        # Gradient dError/dW_ho = outer(hidden, output_error)
        # Gradient dError/dB_o = output_error
        self.weights_ho += learning_rate * np.outer(hidden_flat, output_errors_flat)
        self.biases_o += learning_rate * output_errors_flat

        # Update hidden layer weights and biases
        # Gradient dError/dW_ih = outer(inputs, hidden_error)
        # Gradient dError/dB_h = hidden_error
        self.weights_ih += learning_rate * np.outer(inputs_flat, hidden_errors_flat)
        self.biases_h += learning_rate * hidden_errors_flat

    def get_best_move(self, inputs: np.ndarray, board: list[list[str | None]], strategy: str = 'greedy', epsilon: float = 0.1) -> int:
        """
        Selects a move using epsilon-greedy strategy.
        Crucially, invalid moves are masked *before* selecting the best one.
        """
        available_moves_indices = [i * BOARD_COL + j for i in range(BOARD_ROW) for j in range(BOARD_COL) if board[i][j] is None]

        if not available_moves_indices:
             print(f"Warning: get_best_move called on a board with no available moves:\n{np.array(board)}")
             # This should ideally not happen if called correctly before game end.
             # Return a dummy value or raise error? Let's return -1 to signal issue.
             return -1

        if strategy == 'exploration' and np.random.rand() < epsilon:
            # Exploration: Choose a random *valid* move
            chosen_index = np.random.choice(available_moves_indices)
        else:
            # Exploitation: Choose the best *valid* move according to the network
            output_q_values = self.forward(inputs)

            # Create a mask for valid moves, setting invalid moves' Q-values to -infinity
            masked_q_values = np.full_like(output_q_values, -np.inf, dtype=np.float64) # Use float64 for -inf
            if available_moves_indices: # Ensure list is not empty
                 masked_q_values[available_moves_indices] = output_q_values[available_moves_indices]

            # Find the index of the maximum Q-value among valid moves
            chosen_index = np.argmax(masked_q_values)

            # Sanity check: If all valid moves had -inf Q-value (unlikely), argmax might return 0.
            # Or if all predicted Q-values for valid moves are identical.
            if chosen_index not in available_moves_indices:
                 # This could happen if all valid Q-values are -inf or identical non-inf.
                 # Fallback to random choice among available moves.
                 # print(f"Warning: Argmax chose {chosen_index}, not in {available_moves_indices}. Q-vals: {output_q_values}. Masked: {masked_q_values}. Falling back to random.")
                 chosen_index = np.random.choice(available_moves_indices)

        return chosen_index


# --- Perspective-Aware Input Function ---
def board_to_input(game_state: list[list[str|None]], current_player_symbol: str) -> np.array:
    """
    Converts board state to network input from the perspective of the current player.
    Channel 0: My pieces (current_player_symbol)
    Channel 1: Opponent's pieces
    """
    board_repr = np.zeros((BOARD_ROW, BOARD_COL, 2), dtype=np.float32)
    opponent_symbol = 'O' if current_player_symbol == 'X' else 'X'

    for r in range(BOARD_ROW):
        for c in range(BOARD_COL):
            if game_state[r][c] == current_player_symbol:
                board_repr[r, c, 0] = 1.0  # My piece
            elif game_state[r][c] == opponent_symbol:
                board_repr[r, c, 1] = 1.0  # Opponent's piece
            # Else: Both channels remain 0 for empty cell

    return board_repr.flatten()


# --- TicTacTor Game Class (Modified for Self-Play Compatibility) ---
class TicTacTor:
    def __init__(self):
        self.game_board = GameBoard() # Uses common.GameBoard
        self.current_player_symbol = 'X' # Start with 'X'

    def reset(self):
        self.game_board.reset_board()
        self.current_player_symbol = 'X'

    def get_board_state(self) -> list[list[str|None]]:
        return self.game_board.board

    def get_current_player(self) -> str:
        return self.current_player_symbol

    def is_valid_move(self, row: int, col: int) -> bool:
         # Check bounds and if cell is empty
        return 0 <= row < BOARD_ROW and 0 <= col < BOARD_COL and self.game_board.board[row][col] is None

    def make_move(self, row: int, col: int):
        if not self.is_valid_move(row, col):
            raise ValueError(f"Invalid move attempted: ({row}, {col}) on board {np.array(self.get_board_state())}")
        self.game_board.board[row][col] = self.current_player_symbol
        # Switch player
        self.current_player_symbol = 'O' if self.current_player_symbol == 'X' else 'X'

    def check_winner(self) -> str | None:
        board = self.game_board.board
        # Check rows, columns, and diagonals (same logic as before)
        for i in range(BOARD_ROW):
            # Rows
            if board[i][0] is not None and board[i][0] == board[i][1] == board[i][2]:
                return board[i][0]
            # Columns (check using index i for column)
            if board[0][i] is not None and board[0][i] == board[1][i] == board[2][i]:
                return board[0][i]
        # Diagonals
        if board[0][0] is not None and board[0][0] == board[1][1] == board[2][2]:
            return board[0][0]
        if board[0][2] is not None and board[0][2] == board[1][1] == board[2][0]:
            return board[0][2]
        return None

    def is_draw(self) -> bool:
        # Check if board is full AND there is no winner
        if self.check_winner() is not None:
            return False # Cannot be a draw if there's a winner
        return all(cell is not None for row in self.game_board.board for cell in row)

    def get_available_moves(self) -> list[tuple[int, int]]:
        return [(r, c) for r in range(BOARD_ROW) for c in range(BOARD_COL) if self.game_board.board[r][c] is None]


# --- Rule-Based Opponent (Slightly Improved for Evaluation) ---
def rule_based_opponent_move(game: TicTacTor) -> tuple[int, int]:
    """Rule-based opponent for evaluation."""
    available_moves = game.get_available_moves()
    if not available_moves:
        raise ValueError("RuleBasedOpponent called with no available moves.")

    board = game.get_board_state()
    my_symbol = game.get_current_player()
    opponent_symbol = 'O' if my_symbol == 'X' else 'X'

     # 1. Win if possible
    for r, c in available_moves:
        board[r][c] = my_symbol
        if game.check_winner() == my_symbol:
            board[r][c] = None # Reset test move
            return (r, c)
        board[r][c] = None # Reset test move

    # 2. Block opponent's win
    for r, c in available_moves:
        board[r][c] = opponent_symbol
        if game.check_winner() == opponent_symbol:
            board[r][c] = None # Reset test move
            return (r, c) # Block here
        board[r][c] = None # Reset test move

    # 3. Take center if available
    center = (BOARD_ROW // 2, BOARD_COL // 2)
    if center in available_moves:
        return center

     # 4. Take opposite corner from opponent
    corners = [(0, 0), (0, BOARD_COL - 1), (BOARD_ROW - 1, 0), (BOARD_ROW - 1, BOARD_COL - 1)]
    opposite_corners = {(0, 0): (2, 2), (0, 2): (2, 0), (2, 0): (0, 2), (2, 2): (0, 0)}
    for r_corn, c_corn in corners:
        if board[r_corn][c_corn] == opponent_symbol:
             opp_r, opp_c = opposite_corners[(r_corn, c_corn)]
             # Check if opposite corner is available in the list of valid moves
             if (opp_r, opp_c) in available_moves:
                  return (opp_r, opp_c)

    # 5. Take any available corner
    for r_corn, c_corn in corners:
        if (r_corn, c_corn) in available_moves:
            return (r_corn, c_corn)

    # 6. Take any available side
    sides = [(0, 1), (1, 0), (1, 2), (2, 1)] # Assuming 3x3
    for r_side, c_side in sides:
         if (r_side, c_side) in available_moves:
              return (r_side, c_side)

    # 7. Fallback: Move randomly (should not be needed in standard TicTacToe if logic above is sound)
    # print("Rule-based opponent falling back to random.")
    return available_moves[np.random.randint(len(available_moves))]


# --- Evaluation Function ---
def evaluate_neural_network(nn: NeuralNetwork, num_games: int = 1000, opponent_type='rule') -> tuple[float, float]:
    """Evaluates NN against a specified opponent."""
    wins = 0
    draws = 0
    losses = 0 # NN 'X' loses

    print(f"Evaluating NN ('X') vs '{opponent_type}' opponent ({num_games} games)...")

    for i in range(num_games):
        if (i+1)%200 == 0: print(f" Eval game {i+1}/{num_games}")
        game = TicTacTor()
        game_over = False
        while not game_over:
            current_player = game.get_current_player()
            board_state = game.get_board_state()

            if current_player == 'X': # NN's turn
                inputs = board_to_input(board_state, current_player)
                # Use greedy strategy (no exploration) during evaluation
                move_index = nn.get_best_move(inputs, board_state, strategy='greedy')
                if move_index == -1: # No valid moves
                    # print(" NN had no moves in eval - game should have ended?")
                    break # End eval game loop
                move = (move_index // BOARD_COL, move_index % BOARD_COL)
            else: # Opponent's turn ('O')
                if opponent_type == 'rule':
                    try:
                         move = rule_based_opponent_move(game)
                    except ValueError: # Opponent has no moves
                         # print(" Rule opponent had no moves in eval.")
                         break
                elif opponent_type == 'random':
                     available = game.get_available_moves()
                     if not available: break
                     move = available[np.random.randint(len(available))]
                else:
                    raise ValueError(f"Unknown opponent type: {opponent_type}")

            try:
                 game.make_move(row=move[0], col=move[1])
            except ValueError as e:
                 print(f" EVAL Error: {e}. Player {current_player}, Move {move}, Board:\n{np.array(board_state)}")
                 # If NN caused error, count as loss? Or just skip game? Let's count as loss.
                 if current_player == 'X': losses += 1
                 else: wins += 1 # Opponent failed, NN wins? Or declare draw? Let's make NN win.
                 game_over = True # Stop this game

            winner = game.check_winner()
            if winner:
                if winner == 'X': wins += 1
                else: losses += 1
                game_over = True
            elif game.is_draw():
                draws += 1
                game_over = True

    win_rate = wins / num_games
    draw_rate = draws / num_games
    loss_rate = losses / num_games
    print(f"Evaluation Result vs '{opponent_type}': Win Rate: {win_rate:.2%}, Draw Rate: {draw_rate:.2%}, Loss Rate: {loss_rate:.2%}")
    return win_rate, draw_rate


# --- Model Save/Load ---
def save_model(nn: NeuralNetwork, filepath: str):
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(nn, f)
        print(f"Model saved to {filepath}")
    except Exception as e:
        print(f"Error saving model to {filepath}: {e}")

def load_model(filepath: str) -> NeuralNetwork | None:
     if not os.path.exists(filepath):
          print(f"Model file not found: {filepath}. Starting fresh.")
          return None
     try:
          with open(filepath, 'rb') as f:
               model = pickle.load(f)
          print(f"Model loaded from {filepath}")
          # Basic validation
          if not isinstance(model, NeuralNetwork):
               print("Warning: Loaded object is not a NeuralNetwork instance.")
               return None
          # Optional: Check if parameters match NN_INPUT/HIDDEN/OUTPUT_SIZE
          # if model.input_size != NN_INPUT_SIZE or ...
          return model
     except Exception as e:
          print(f"Error loading model from {filepath}: {e}")
          return None


# --- Training Function with Self-Play ---
def train_neural_network_selfplay(
    nn: NeuralNetwork,
    episodes: int = 100000,  # Adjust number of episodes
    learning_rate: float = 0.001, # Slightly higher LR might work with self-play
    discount_factor: float = 0.95, # Discount factor gamma
    epsilon_start: float = 0.9, # Start with high exploration
    epsilon_end: float = 0.01, # Minimum exploration
    epsilon_decay_steps: int = 50000, # Steps over which epsilon decays
    patience: int = 15,
    min_delta: float = 0.01, # Min win rate improvement to reset patience
    model_save_path: str = 'best_selfplay_model.pkl',
    eval_opponent_type: str = 'rule', # Evaluate against rules
    eval_frequency: int = 2000, # How often to evaluate
    print_frequency: int = 500
):
    """Trains the neural network using self-play."""

    best_eval_win_rate = -1.0 # Track best win rate against eval opponent
    no_improvement_count = 0
    epsilon = epsilon_start
    # Calculate decay factor per episode based on total steps
    # Alternative: decay per step/move if preferred
    if epsilon_decay_steps > 0:
         epsilon_decay_factor = np.exp(np.log(epsilon_end / epsilon_start) / epsilon_decay_steps)
    else:
         epsilon_decay_factor = 1.0 # No decay if steps is 0

    print(f"Starting self-play training for {episodes} episodes...")
    print(f"Params: LR={learning_rate}, Gamma={discount_factor}, EpsilonDecayFactor={epsilon_decay_factor:.6f} (over {epsilon_decay_steps} episodes)")
    print(f"Saving best model (vs '{eval_opponent_type}') to: {model_save_path}")

    loaded_nn = load_model(model_save_path)
    if loaded_nn:
        nn = loaded_nn
        # Consider loading best_eval_win_rate if saved separately

    for episode in range(episodes):
        game = TicTacTor()
        # History stores tuples: (player_symbol, state_input, action_index)
        history = []
        game_over = False
        winner = None
        move_count = 0

        while not game_over:
            current_player = game.get_current_player()
            board_state = game.get_board_state()

            # Get perspective-aware input for the NN
            inputs = board_to_input(board_state, current_player)

            # Both players use the NN with epsilon-greedy exploration
            move_index = nn.get_best_move(inputs, board_state, strategy='exploration', epsilon=epsilon)

            if move_index == -1: # Should not happen in a valid game state
                print(f"ERROR: NN returned no valid move for {current_player} in episode {episode+1}. Board:\n{np.array(board_state)}")
                break # End this problematic episode

            # Store state, action, and who took it
            history.append((current_player, inputs, move_index))

            # Perform the move
            move = (move_index // BOARD_COL, move_index % BOARD_COL)
            try:
                game.make_move(row=move[0], col=move[1])
                move_count +=1
            except ValueError as e:
                print(f"ERROR during game.make_move in episode {episode+1}: {e}")
                print(f"Player: {current_player}, Proposed Move: {move}, Index: {move_index}")
                print(f"Board State:\n{np.array(board_state)}")
                game_over = True # End problematic episode
                continue # Skip reward/backprop for this episode

            # Check for game end
            winner = game.check_winner()
            if winner:
                game_over = True
            elif game.is_draw():
                game_over = True

        # --- Episode End: Assign Reward and Backpropagate ---
        if game_over: # Ensure game actually finished before processing
            final_reward = 0.0
            if winner == 'X':
                final_reward = 1.0
            elif winner == 'O':
                final_reward = -1.0
            # final_reward remains 0.0 for a draw

            # Backpropagate through the history
            num_moves_in_history = len(history)
            for i, (player_symbol, state_input, action_index) in enumerate(reversed(history)):
                # Calculate discounted reward G_t = reward * gamma^(n-1-i) where n=total_moves
                # Alternatively: G_t = reward * gamma^k where k is steps from end (i)
                discounted_reward = final_reward * (discount_factor ** i)

                # Adjust reward based on the player who made the move
                # If player 'O' made the move, their reward perception is inverse of final_reward
                if player_symbol == 'O':
                    target_value = -discounted_reward
                else: # Player 'X' made the move
                    target_value = discounted_reward

                # --- Prepare targets for backpropagation (Q-learning style target) ---
                # Get the network's prediction for the state encountered *by that player*
                predicted_outputs = nn.forward(state_input) # Input was already perspective-aware
                # Create the target vector
                targets = predicted_outputs.copy()
                # Set the target for the action *actually taken* in that state
                targets[action_index] = target_value # Directly set the target Q-value estimate

                # --- Perform backpropagation ---
                nn.backward(targets, learning_rate)

        # --- Epsilon Decay (per episode) ---
        epsilon = max(epsilon_end, epsilon * epsilon_decay_factor)
        # Alternative: decay per step/move_count if game lengths vary wildly

        # --- Logging ---
        if (episode + 1) % print_frequency == 0:
            print(f"Ep {episode + 1}/{episodes} | Epsilon: {epsilon:.4f} | Moves: {move_count} | Winner: {winner if winner else 'Draw'}")

        # --- Evaluation and Early Stopping ---
        if (episode + 1) % eval_frequency == 0:
            print(f"\n--- Evaluating model at episode {episode + 1} ---")
            current_win_rate, current_draw_rate = evaluate_neural_network(nn, num_games=300, opponent_type=eval_opponent_type) # More games for stability

            if current_win_rate > best_eval_win_rate + min_delta:
                print(f"Improvement! Eval Win Rate vs '{eval_opponent_type}': {current_win_rate:.2%} > Best: {best_eval_win_rate:.2%}")
                best_eval_win_rate = current_win_rate
                no_improvement_count = 0
                save_model(nn, model_save_path)
            else:
                no_improvement_count += 1
                print(f"No significant improvement. Eval Win Rate: {current_win_rate:.2%}, Best: {best_eval_win_rate:.2%}. No improvement count: {no_improvement_count}/{patience}")

            if no_improvement_count >= patience:
                print(f"\nEarly stopping triggered after {patience} evaluations with no improvement.")
                break
            print("--- Evaluation complete ---\n")

    print(f"\nSelf-play training finished after {episode + 1} episodes.")
    # Final evaluation
    print("--- Final Model Evaluation ---")
    evaluate_neural_network(nn, num_games=1000, opponent_type='rule')
    evaluate_neural_network(nn, num_games=1000, opponent_type='random')


# --- Main Execution ---
if __name__ == "__main__":
     # Initialize network
     nn = NeuralNetwork(input_size=NN_INPUT_SIZE, hidden_size=NN_HIDDEN_SIZE, output_size=NN_OUTPUT_SIZE)
     print(f"Neural Network Initialized: Input={nn.input_size}, Hidden={nn.hidden_size}, Output={nn.output_size}")

     # Start self-play training
     train_neural_network_selfplay(
        nn,
        episodes=200000,      # Number of self-play games
        learning_rate=0.0005, # May need tuning
        discount_factor=0.97, # High discount for short game
        epsilon_start=1.0,    # Start exploring fully
        epsilon_end=0.01,
        epsilon_decay_steps=100000, # Decay over this many episodes
        patience=20,          # More patience for self-play fluctuations
        min_delta=0.01,       # 1% win rate improvement needed
        model_save_path='tic_tac_toe_selfplay_nn_v1.pkl',
        eval_opponent_type='rule', # Evaluate against rule-based
        eval_frequency=5000,   # Evaluate every 5k episodes
        print_frequency=1000
     )

     print("\nTraining script finished.")
     # Example of loading and evaluating the best saved model again
     print("\nLoading best saved model for final check...")
     best_model = load_model('tic_tac_toe_selfplay_nn_v1.pkl')
     if best_model:
          evaluate_neural_network(best_model, num_games=2000, opponent_type='rule')
     else:
          print("Could not load saved model.")
