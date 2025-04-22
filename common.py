BOARD_ROW = 3
BOARD_COL = 3
BOARD_SIZE = BOARD_ROW * BOARD_COL

class GameBoard:
    
    board: list[list[str|None]]
    current_player: int  # 0 for AI(X), 1 for human(O)
    col: int  # 3
    row: int  # 3
    board_size: int  # 9
    
    def __init__(self, col: int = BOARD_COL, row: int = BOARD_ROW):
        self.board_size = BOARD_SIZE
        self.col = col
        self.row = row
        self.board = [[None for _ in range(self.col)] for _ in range(self.row)]
        self.current_player = 0

    def reset_board(self):
        self.current_player = 0
        self.board = [[None for _ in range(self.col)] for _ in range(self.row)]

