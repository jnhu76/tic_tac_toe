import numpy as np
from common import GameBoard, BOARD_SIZE, BOARD_COL, BOARD_ROW
import pickle


NN_INPUT_SIZE = 18
NN_HIDDEN_SIZE = 100
NN_OUTPUT_SIZE = 9


class NeuralNetwork:
    def __init__(self, input_size: int = NN_INPUT_SIZE, hidden_size: int = NN_HIDDEN_SIZE, output_size: int = NN_OUTPUT_SIZE):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始化权重和偏置
        self.weights_ih = np.random.randn(input_size, self.hidden_size) * np.sqrt(2 / input_size)
        self.weights_ho = np.random.randn(self.hidden_size, output_size) * np.sqrt(2 / self.hidden_size)
        self.biases_h = np.zeros(self.hidden_size)
        self.biases_o = np.zeros(output_size)
        
        # 初始化激活值
        self.inputs = np.zeros(input_size)
        self.hidden = np.zeros(self.hidden_size)
        self.raw_logits = np.zeros(output_size)
        self.outputs = np.zeros(output_size)
    
    def leaky_relu(self, x: np.array, alpha: float = 0.01) -> np.array:
        return np.where(x > 0, x, x * alpha)
    
    def leaky_relu_derivative(self, x: np.array, alpha: float = 0.01) -> np.array:
        return np.where(x > 0, 1, alpha)

    def forward(self, inputs: np.array) -> np.array:
        self.inputs = inputs
        
        # 计算隐藏层激活值
        self.hidden = np.dot(self.inputs, self.weights_ih) + self.biases_h
        self.hidden = self.leaky_relu(self.hidden)
        
        # 计算输出层激活值
        self.outputs = np.dot(self.hidden, self.weights_ho) + self.biases_o
        self.outputs = self.softmax(self.outputs)
        
        return self.outputs
    
    def sigmoid(self, x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x: np.array) -> np.array:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def relu_derivative(self, x: np.array) -> np.array:
        return (x > 0).astype(float)

    def backward(self, targets: np.array, learning_rate: float, reward_scaling: float = 1.0):
        """
        反向传播更新权重和偏置。
        
        参数:
            targets (np.ndarray): 目标值（期望输出）。
            learning_rate (float): 学习率。
            reward_scaling (float): 奖励缩放因子。
        """
        # 计算输出层误差
        output_errors = (targets - self.outputs) * reward_scaling # 直接使用 reward，不要取绝对值，符号很重要！
        # 确保 output_errors 的形状为 (output_size,)
        output_errors = output_errors.reshape(self.output_size)
        # 计算隐藏层误差
        hidden_errors = np.dot(output_errors, self.weights_ho.T)
        # *** 使用正确的导数 ***
        hidden_errors = hidden_errors * self.leaky_relu_derivative(self.hidden) # <--- 修改这里
        # ... (weight/bias updates remain the same)
        # 更新权重和偏置 (使用修正后的误差)
        self.weights_ho += learning_rate * np.outer(self.hidden, output_errors)
        self.biases_o += learning_rate * output_errors # 确保 output_errors 是一维的
        # 确保 inputs 和 hidden_errors 形状匹配以便外积
        if self.inputs.ndim == 1 and hidden_errors.ndim == 1:
            self.weights_ih += learning_rate * np.outer(self.inputs, hidden_errors)
        else:
            # Handle potential dimension issues or skip update if shapes are wrong
            print(f"Warning: Skipping weights_ih update due to shape mismatch. Inputs: {self.inputs.shape}, Hidden Errors: {hidden_errors.shape}")
            # Or potentially reshape: self.weights_ih += learning_rate * np.outer(self.inputs.flatten(), hidden_errors.flatten()) - check dimensions carefully
        self.biases_h += learning_rate * hidden_errors # 确保 hidden_errors 是一维的
        
    def calculate_total_parameters(self) -> int:
        # 计算输入层到隐藏层的权重参数数量
        params_ih = self.input_size * self.hidden_size
        
        # 计算隐藏层到输出层的权重参数数量
        params_ho = self.hidden_size * self.output_size
        
        # 计算隐藏层的偏置参数数量
        params_bh = self.hidden_size
        
        # 计算输出层的偏置参数数量
        params_bo = self.output_size
        
        # 总参数量
        total_params = params_ih + params_ho + params_bh + params_bo
        return total_params
    
    def relu(self, x: np.array) -> np.array:
        return np.maximum(0, x)
        
    def get_best_move(self, inputs: np.array, board: list[list[str | None]], strategy: str = 'greedy', epsilon: float = 0.1) -> int:
        """
        根据神经网络的输出计算最优位置。
        
        参数:
            inputs (np.ndarray): 输入状态（18 维向量）。
            board (list[list[str|None]]): 当前棋盘状态。
            strategy (str): 选择策略，可选值为 "greedy" 或 "exploration"。
            epsilon (float): ε-greedy 策略中的探索概率。
        
        返回:
            int: 最优位置的索引（0 到 8）。
        """
        if strategy == 'greedy':
            # 贪心策略，选择输出值最大的有效位置
            output = self.forward(inputs)
            available_moves = [i * BOARD_COL + j for i in range(BOARD_ROW) for j in range(BOARD_COL) if board[i][j] is None]
            valid_output = np.zeros_like(output)
            valid_output[available_moves] = output[available_moves]
            best_move = np.argmax(valid_output)
        elif strategy == 'exploration':
            # ε-greedy 策略
            if np.random.rand() < epsilon:
                # 随机选择一个合法动作
                available_moves = [(i, j) for i in range(BOARD_ROW) for j in range(BOARD_COL) if board[i][j] is None]
                if not available_moves:
                    raise ValueError("No available moves to choose from.")
                chosen_move = available_moves[np.random.randint(len(available_moves))]
                best_move = chosen_move[0] * BOARD_COL + chosen_move[1]  # 将坐标转换为索引
            else:
                # 贪心选择
                output = self.forward(inputs)
                available_moves = [i * BOARD_COL + j for i in range(BOARD_ROW) for j in range(BOARD_COL) if board[i][j] is None]
                valid_output = np.zeros_like(output)
                valid_output[available_moves] = output[available_moves]
                best_move = np.argmax(valid_output)
        else:
            raise ValueError("Invalid strategy. Choose 'greedy' or 'exploration'.")
        
        return best_move


def board_to_input(GameState: list[list[str|None]], col: int = 3, row: int = 3) -> np.array:
    # 将棋盘状态转换为神经网络的输入
    # 这里可以根据具体的棋盘表示方式进行转换
    # None: empty, 00;  'X' or 'x': player(x), 10;  'O' or 'o': player(o), 01
    board = np.zeros((col, row, 2))
    
    for i in range(col):
        for j in range(row):
            if GameState[i][j] == 'X' or GameState[i][j] == 'x':
                board[i, j, 0] = 1
                board[i, j, 1] = 0
            elif GameState[i][j] == 'O' or GameState[i][j] == 'o':
                board[i, j, 0] = 0
                board[i, j, 1] = 1
            else:
                board[i, j, 0] = 0
                board[i, j, 1] = 0
    
    return board.flatten()


# 初始化神经网路

nn = NeuralNetwork(input_size=NN_INPUT_SIZE, hidden_size=NN_HIDDEN_SIZE, output_size=NN_OUTPUT_SIZE)

# 训练神经网络

class TicTacTor:
    
    def __init__(self):
        self.game_board = GameBoard()
        self.current_player = 'X'
        
    def is_valid_move(self, row: int, col: int) -> bool:
        return self.game_board.board[row][col] is None
    
    def make_move(self, row:int, col:int):
        if not self.is_valid_move(row, col):
            # print(f"Invalid move: ({row}, {col}), Board: {np.array(self.game_board.board)}")
            raise ValueError("Invalid move")
        self.game_board.board[row][col] = self.current_player
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        
    def is_draw(self):
        return all(cell is not None for row in self.game_board.board for cell in row)
    
    def check_winner(self) -> str | None:
        # 检查行、列、对角线是否有获胜者
        for i in range(BOARD_ROW):
            if self.game_board.board[i][0] == self.game_board.board[i][1] == self.game_board.board[i][2] and self.game_board.board[i][0]:
                return self.game_board.board[i][0]
            if self.game_board.board[0][i] == self.game_board.board[1][i] == self.game_board.board[2][i] and self.game_board.board[0][i]:
                return self.game_board.board[0][i]
        if self.game_board.board[0][0] == self.game_board.board[1][1] == self.game_board.board[2][2] and self.game_board.board[0][0]:
            return self.game_board.board[0][0]
        if self.game_board.board[0][2] == self.game_board.board[1][1] == self.game_board.board[2][0] and self.game_board.board[0][2]:
            return self.game_board.board[0][2]
        return None
    
    def reset(self):
        self.game_board.reset_board()
        self.current_player = 'X'
        
    def get_board(self):
        return self.game_board.board
    
    def get_current_player(self):
        return self.current_player
    
    def get_board_size(self):
        return self.game_board.board_size
    
    def get_board_row(self):
        return self.game_board.row
    
    def get_board_col(self):
        return self.game_board.col
    
    def get_board_state(self):
        return self.game_board.board

    def get_board_state_flatten(self):
        return np.array(self.game_board.board).flatten()
    
    def get_available_moves(self):
        available_moves = []
        for i in range(self.game_board.row):
            for j in range(self.game_board.col):
                if self.game_board.board[i][j] is None:
                    available_moves.append((i, j))
        return available_moves
    

def random_opponent_move(game: TicTacTor) -> tuple[int, int]:
    """
    随机选择一个合法动作。
    
    参数:
        game (TicTacTor): 当前游戏实例。
    
    返回:
        tuple[int, int]: 随机选择的动作 (row, col)。
    """
    available_moves = game.get_available_moves()
    if not available_moves:
        raise ValueError("No available moves to choose from.")
    return available_moves[np.random.randint(len(available_moves))]


def rule_based_opponent_move(game: TicTacTor) -> tuple[int, int]:
    board = game.get_board_state()
    center = (1, 1)
    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    available_moves = game.get_available_moves()
    
    # 优先占据中心
    if center in available_moves:
        return center
    
    # 优先占据角落
    for corner in corners:
        if corner in available_moves:
            return corner
    
    # 随机选择其他位置
    return available_moves[np.random.randint(len(available_moves))]

def train_neural_network(
    nn: NeuralNetwork, 
    episodes: int = 2000000,  # 增加训练轮数
    learning_rate: float = 0.0005,  # 调整学习率
    discount_factor: float = 0.97,  # 调整折扣因子
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.999,  # 调整epsilon_decay值，使衰减更慢
    patience: int = 10,  
    min_delta: float = 0.01,  
    model_save_path: str = 'best_model.pkl',  
    epsilon: float = 0.1  
):
    best_win_rate = 0
    no_improvement_count = 0
    epsilon_current = epsilon_start  # 修改变量名以避免冲突

    for episode in range(episodes):
        game = TicTacTor()
        history = []
        winner = None
        
        while True:
            # 检查是否已有获胜者
            winner = game.check_winner()
            if winner:
                break
                
            if game.current_player == 'X' or game.current_player == 'x':
                inputs = board_to_input(game.get_board_state())
                move_index = nn.get_best_move(inputs, game.get_board_state(), strategy='exploration', epsilon=epsilon_current)
                
                # 检查 move_index 是否有效
                if move_index < 0 or move_index >= BOARD_SIZE:
                    raise ValueError(f"Invalid move_index: {move_index}")
                
                move = (move_index // BOARD_COL, move_index % BOARD_COL)
                history.append((inputs, move_index))
            else:
                move = rule_based_opponent_move(game)  # 使用规则型对手
            
            # 更新棋盘状态
            try:
                if not game.is_valid_move(move[0], move[1]):
                    print(f"Invalid move detected: {move}, Board: {np.array(game.get_board_state())}")
                    raise ValueError("Invalid move")
                game.make_move(row=move[0], col=move[1])
            except ValueError as e:
                print(f"Error: {e}")
                print(f"Move: {move}, Board: {np.array(game.get_board_state())}")
                raise
            
            # 检查是否平局
            if game.is_draw():
                break
        
        # 根据游戏结果更新奖励
        if winner == 'X' or winner == 'x':
            reward = 1.0
        elif winner == 'O' or winner == 'o':
            reward = -1.0
        else:
            reward = 0.0
        
        # 增加中间奖励
        for move_idx, (inputs, move_index) in enumerate(reversed(history)):
            row, col = move_index // BOARD_COL, move_index % BOARD_COL
            discounted_reward = reward * (discount_factor ** move_idx)
            
            # 鼓励占据中心和潜在连线
            if (row, col) == (1, 1):  # 中心位置
                discounted_reward += 0.3  # 提高中心位置奖励
            elif row == col or row + col == 2:  # 对角线位置
                discounted_reward += 0.2  # 提高对角线位置奖励
            
            # 新增：鼓励潜在连线（两子连成一线）
            board_state = game.get_board_state()
            if sum(1 for i in range(BOARD_ROW) if board_state[i][col] == 'X') == 2:  # 列连线
                discounted_reward += 0.4
            if sum(1 for j in range(BOARD_COL) if board_state[row][j] == 'X') == 2:  # 行连线
                discounted_reward += 0.4
            if sum(1 for i in range(BOARD_ROW) if board_state[i][i] == 'X') == 2:  # 主对角线连线
                discounted_reward += 0.4
            if sum(1 for i in range(BOARD_ROW) if board_state[i][BOARD_ROW - 1 - i] == 'X') == 2:  # 副对角线连线
                discounted_reward += 0.4
            
            # 新增：鼓励占据边角位置
            if (row, col) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                discounted_reward += 0.2  # 提高边角位置奖励
            
            targets = nn.forward(inputs)
            targets[move_index] += discounted_reward
            nn.backward(targets, learning_rate)
        
        # 动态调整ε值
        epsilon_current = max(epsilon_end, epsilon_current * epsilon_decay)
        
        # 打印进度
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{episodes} completed, Epsilon: {epsilon_current:.4f}")
            
        # 定期评估模型
        if (episode + 1) % 10000 == 0:
            win_rate, _ = evaluate_neural_network(nn, num_games=100)
            
            # 检查是否需要早停
            if win_rate > best_win_rate + min_delta:
                best_win_rate = win_rate
                no_improvement_count = 0  
                
                # 保存最佳模型
                save_model(nn, model_save_path)
                print(f"New best model saved with Win Rate: {best_win_rate:.2%}")
            else:
                no_improvement_count += 1  
            
            print(f"Current Win Rate: {win_rate:.2%}, Best Win Rate: {best_win_rate:.2%}")
            
            if no_improvement_count >= patience:
                print("Early stopping triggered.")
                break


def evaluate_neural_network(nn: NeuralNetwork, num_games: int = 1000) -> tuple[float, float]:
    wins = 0
    draws = 0
    
    for _ in range(num_games):
        game = TicTacTor()
        while True:
            # 神经网络玩家
            if game.current_player == 'X':
                inputs = board_to_input(game.get_board_state())
                move_index = nn.get_best_move(inputs, game.get_board_state(), strategy='greedy')
                move = (move_index // BOARD_COL, move_index % BOARD_COL)
            else:
                # 随机对手
                move = random_opponent_move(game)
            
            game.make_move(row=move[0], col=move[1])
            
            winner = game.check_winner()
            if winner:
                if winner == 'X':
                    wins += 1
                break
            elif game.is_draw():
                draws += 1
                break
    
    win_rate = wins / num_games
    draw_rate = draws / num_games
    print(f"Win Rate: {win_rate:.2%}, Draw Rate: {draw_rate:.2%}")
    return win_rate, draw_rate


def save_model(nn: NeuralNetwork, filepath: str):
    with open(filepath, 'wb') as f:
        pickle.dump(nn, f)
        
def load_model(filepath: str) -> NeuralNetwork:
    with open(filepath, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    train_neural_network(nn, episodes=2000000, learning_rate=0.005, discount_factor=0.97, epsilon=0.1, patience=5, min_delta=0.01)
