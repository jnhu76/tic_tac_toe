from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.properties import ObjectProperty
from common import GameBoard
import pickle
import numpy as np
from train import NeuralNetwork  # 导入 NeuralNetwork 类

class AIManager:
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)
    
    def load_model(self, model_path: str) -> NeuralNetwork:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def predict_move(self, board: list[list[str | None]]) -> int:
        inputs = self.board_to_input(board)
        move_index = self.model.get_best_move(inputs, board, strategy='greedy')
        return move_index
    
    def board_to_input(self, GameState: list[list[str | None]], col: int = 3, row: int = 3) -> np.array:
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

class GameGrid(GridLayout):
    def __init__(self, **kwargs):
        super(GameGrid, self).__init__(**kwargs)
        self.cols = 3
        self.rows = 3
        self.board = GameBoard()
        self.buttons = []
        
        for i in range(self.board.board_size):
            btn = Button(text='', font_size=40)
            btn.bind(on_press=self.on_button_press)
            self.add_widget(btn)
            self.buttons.append(btn)
        
        # 初始化 AI 管理器
        self.ai_manager = AIManager('best_model.pkl')
        # 移除默认的ai_move()调用

    def on_button_press(self, instance):
        index = self.buttons.index(instance)
        if instance.text == '' and self.board.current_player == 1:
            instance.text = 'O'
            self.board.board[index // 3][index % 3] = 'O'
            self.board.current_player = 0
            if not self.check_game_status():  # 只有在游戏未结束时才让AI下棋
                self.ai_move()

    def ai_move(self):
        if self.board.current_player == 0:
            # 检查是否已有获胜者
            winner = self.check_game_status()
            if not winner:
                move = self.ai_manager.predict_move(self.board.board)
                if move is not None:
                    row, col = move // 3, move % 3
                    self.buttons[move].text = 'X'
                    self.board.board[row][col] = 'X'
                    self.board.current_player = 1
                self.check_game_status()

    def check_game_status(self):
        # 检查行、列、对角线是否有获胜者
        for i in range(3):
            if self.board.board[i][0] == self.board.board[i][1] == self.board.board[i][2] and self.board.board[i][0] is not None:
                self.game_over(self.board.board[i][0])
                return True
            if self.board.board[0][i] == self.board.board[1][i] == self.board.board[2][i] and self.board.board[0][i] is not None:
                self.game_over(self.board.board[0][i])
                return True
        if self.board.board[0][0] == self.board.board[1][1] == self.board.board[2][2] and self.board.board[0][0] is not None:
            self.game_over(self.board.board[0][0])
            return True
        if self.board.board[0][2] == self.board.board[1][1] == self.board.board[2][0] and self.board.board[0][2] is not None:
            self.game_over(self.board.board[0][2])
            return True
        
        # 检查是否平局
        if all(cell is not None for row in self.board.board for cell in row):
            self.game_over('Draw')
            return True
        
        return False

    def game_over(self, winner):
        if winner == 'X':
            self.parent.status_label.text = "人类你输了"
        elif winner == 'O':
            self.parent.status_label.text = "人类你赢了"
        else:
            self.parent.status_label.text = "平局"
        self.disable_buttons()

    def disable_buttons(self):
        for btn in self.buttons:
            btn.disabled = True

    def reset_game(self):
        self.board.reset_board()
        for btn in self.buttons:
            btn.text = ''
            btn.disabled = False
        self.parent.status_label.text = "游戏开始"
        self.ai_move()

class GameControls(BoxLayout):
    status_label = ObjectProperty(None)
    game_grid = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(GameControls, self).__init__(**kwargs)
        self.orientation = 'horizontal'  # 设置水平布局
        
        # 添加背景颜色
        with self.canvas.before:
            from kivy.graphics import Color, Rectangle
            Color(0.9, 0.9, 0.9, 1)  # 浅灰色背景
            self.rect = Rectangle(size=self.size, pos=self.pos)
        
        # 绑定尺寸变化以动态调整背景
        self.bind(size=self._update_rect, pos=self._update_rect)
        
        # 添加左侧游戏网格
        self.game_grid = GameGrid()
        self.add_widget(self.game_grid)
        
        # 添加右侧控制面板
        control_panel = BoxLayout(orientation='vertical', size_hint=(0.3, 1))
        control_panel.add_widget(Label(
            text='[b]井字棋游戏[/b]', 
            markup=True, 
            font_size=24,
            font_name='assets/fonts/SourceHanSansSC-Normal-2.otf',  # 指定中文字体
            color=[0, 0, 0, 1]  # 字体颜色改为黑色
        ))  # 标题
        self.status_label = Label(
            text="游戏开始", 
            size_hint=(1, 0.2),
            font_name='assets/fonts/SourceHanSansSC-Normal-2.otf',  # 指定中文字体
            color=[0, 0, 0, 1]  # 字体颜色改为黑色
        )  # 状态显示区域
        control_panel.add_widget(self.status_label)
        
        # 添加按钮
        start_button = Button(
            text="开始游戏", 
            size_hint=(1, 0.2),
            font_name='assets/fonts/SourceHanSansSC-Normal-2.otf',  # 指定中文字体
            background_color=[1, 1, 1, 1],  # 背景颜色改为白色
            color=[0, 0, 0, 1]  # 字体颜色改为黑色
        )
        start_button.bind(on_press=lambda _: self.start_game())
        control_panel.add_widget(start_button)
        
        reset_button = Button(
            text="重置游戏", 
            size_hint=(1, 0.2),
            font_name='assets/fonts/SourceHanSansSC-Normal-2.otf',  # 指定中文字体
            background_color=[1, 1, 1, 1],  # 背景颜色改为白色
            color=[0, 0, 0, 1]  # 字体颜色改为黑色
        )
        reset_button.bind(on_press=lambda _: self.reset_game())
        control_panel.add_widget(reset_button)
        
        self.add_widget(control_panel)  # 将控制面板添加到主布局

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size
    
    def start_game(self):
        self.game_grid.reset_game()
        # 确保 AI 在点击“开始游戏”后才下棋
        if self.game_grid.board.current_player == 0:
            self.game_grid.ai_move()

    def reset_game(self):
        self.game_grid.reset_game()

class TicTacToeApp(App):
    def build(self):
        return GameControls()

if __name__ == '__main__':
    TicTacToeApp().run()