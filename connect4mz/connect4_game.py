import sys
import time

import pygame as pg
import numpy as np


class Connect4:
    def __init__(self, rows: int = 6, cols: int = 7, num_to_win: int = 4, headless: bool = False):
        self.rows = rows
        self.cols = cols
        self.screen = pg.display.set_mode((self.cols * 100, self.rows * 100)) if not headless else None
        self.num_to_win = num_to_win
        self.board = np.zeros((self.rows, self.cols))

    def play(self, col: int, player: int, board: np.ndarray = None):
        if board is None:
            board = self.board.copy()
        column_zeros = np.where(board[:, col] == 0)
        if len(column_zeros[0]) == 0:
            raise ValueError("Column is full")
        top_most = column_zeros[0][-1]
        board[top_most, col] = player
        self.board = board.copy()
        return board

    def draw_grid(self, color: str = "white"):
        for i in range(self.rows):
            for j in range(self.cols):
                pg.draw.rect(self.screen, color, (j * 100, i * 100, 100, 100), 2)

    def draw_board(self, color1: str = "blue", color2: str = "red", board: np.ndarray = None):
        if board is None:
            board = self.board.copy()
        for i in range(self.rows):
            for j in range(self.cols):
                if board[i, j] == 1:
                    pg.draw.circle(self.screen, color1, (j * 100 + 50, i * 100 + 50), 40)
                elif board[i, j] == -1:
                    pg.draw.circle(self.screen, color2, (j * 100 + 50, i * 100 + 50), 40)

    def render(self, board: np.ndarray = None):
        if self.screen is None:
            return
        if board is None:
            board = self.board.copy()
        self.screen.fill((0, 0, 0))
        self.draw_grid()
        self.draw_board(board=board)
        pg.display.update()

    def check_win(self, player: int, board: np.ndarray):
        """
        Check if player has won the game. This method essentially performs a 2d convolution (without the summing) and on each stride, check if the player has won.
        This makes the method player invariant and board size invariant as well as efficient.
        :param player:
        :param board:
        :return:
        """
        w = np.ones((self.num_to_win, self.num_to_win))
        # + 1 because the start position of the filter also counts.
        strides = ((self.rows - self.num_to_win) + 1) * ((self.cols - self.num_to_win) + 1)
        player_board = (board == player).astype(int)
        for i in range(strides):
            row = i // ((self.cols - self.num_to_win) + 1)
            col = i % ((self.cols - self.num_to_win) + 1)
            res = player_board[row:row + self.num_to_win, col:col + self.num_to_win] * w
            if np.any(np.sum(res, axis=0) == self.num_to_win):
                return True
            if np.any(np.sum(res, axis=1) == self.num_to_win):
                return True
            if np.sum(np.diag(res)) == self.num_to_win or np.sum(
                    np.diag(np.fliplr(res))) == self.num_to_win:
                return True
        return False

    def play_random(self):
        player = -1
        while True:
            col = np.random.randint(0, self.cols)
            try:
                board = self.play(col, player)
            except ValueError:
                continue
            if self.check_win(player, board):
                print(f"Player {player} wins!")
                print(board)
                return player

            time.sleep(1)
            self.render()
            player = -player

    def game_result(self, player: int, board: np.ndarray = None):
        if board is None:
            board = self.board.copy()
        if self.check_win(player, board):
            return 1.0
        elif self.check_win(-player, board):
            return -1.0
        elif np.all(board != 0):
            return 1e-4
        return None

    def step(self, action: int, player: int, board: np.ndarray = None):
        if board is None:
            board = self.board.copy()
        reward = 0  # implement your own reward handling if you need rewards.
        board = self.play(action, player, board)
        done = self.game_result(player, board) is not None
        return board, reward, done

    def get_human_input(self, board: np.ndarray):
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.display.quit()
                    pg.quit()
                    sys.exit(0)

            if pg.mouse.get_pressed()[0]:
                x, y = (x // 100 for x in pg.mouse.get_pos())
                if board[y][x][0] == 0:
                    # time.sleep(1)
                    return y, x


