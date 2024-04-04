import io
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
import seaborn as sns
import torch as th
from PIL import Image

from mu_alpha_zero.General.az_game import AlphaZeroGame


class TicTacToeGameManager(AlphaZeroGame):
    """
    This class is the game manager for the game of Tic Tac Toe and its variants.
    """

    def __init__(self, board_size: int, headless: bool, num_to_win=None) -> None:
        # TODO: Implement the possibility to play over internet using sockets.
        self.player = 1
        self.enemy_player = -1
        self.board_size = board_size
        self.board = self.initialize_board()
        self.headless = headless
        self.num_to_win = self.init_num_to_win(num_to_win)
        self.screen = self.create_screen(headless)

    def play(self, player: int, index: tuple) -> None:
        self.board[index] = player

    def init_num_to_win(self, num_to_win: int | None) -> int:
        if num_to_win is None:
            num_to_win = self.board_size
        if num_to_win > self.board_size:
            raise Exception("Num to win can't be greater than board size")
        return num_to_win

    def initialize_board(self):
        board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        return board

    def get_random_valid_action(self, observations: np.ndarray) -> list:
        valid_moves = self.get_valid_moves(observations)
        if len(valid_moves) == 0:
            raise Exception("No valid moves")
        return random.choice(valid_moves)

    def create_screen(self, headless):
        if headless:
            return

        pg.init()
        pg.display.set_caption("Tic Tac Toe")
        board_rect_size = self.board_size * 100
        screen = pg.display.set_mode((board_rect_size, board_rect_size))
        return screen

    def full_iterate_array(self, arr: np.ndarray, func: callable) -> list:
        """
        This function iterates over all rows, columns and the two main diagonals of supplied array,
        applies the supplied function to each of them and returns the results in a list.
        :param arr: a 2D numpy array.
        :param func: a callable function that takes a 1D array as input and returns a result.
        :return: A list of results.
        """
        results = []
        results.append("row")
        for row in arr:
            results.append(func(row.reshape(-1)))

        results.append("col")
        for col in arr.T:
            results.append(func(col.reshape(-1)))

        diags = [np.diag(arr, k=i) for i in range(-arr.shape[0] + 1, arr.shape[1])]
        flipped_diags = [np.diag(np.fliplr(arr), k=i) for i in range(-arr.shape[0] + 1, arr.shape[1])]
        diags.extend(flipped_diags)
        for idx, diag in enumerate(diags):
            if idx == 0:
                results.append("diag_left")
            elif idx == len(diags) // 2:
                results.append("diag_right")
            results.append(func(diag.reshape(-1)))

        # results.append("diag_left")
        # results.append(func(arr.diagonal().reshape(-1)))
        # results.append("diag_right")
        # results.append(func(np.fliplr(arr).diagonal().reshape(-1)))
        return results

    def eval_board(self, board: np.ndarray,check_end: bool = True) -> int | None:
        if self.is_board_full(board):
            return 0
        score = 0
        for i in range(self.num_to_win, self.num_to_win // 2, -1):
            current_won = self.check_partial_win(-1, self.num_to_win, board=board)
            opp_won = self.check_partial_win(1, self.num_to_win, board=board)
            if current_won:
                score += 1 * 2
            if opp_won:
                score -= 1
        if check_end:
            return None if self.game_result(-1, board) is None else score
        return score

    def full_iterate_array_all_diags(self, arr: np.ndarray, func: callable):

        results = []
        for row in arr:
            results.append(func(row.reshape(-1)))

        for col in arr.T:
            results.append(func(col.reshape(-1)))

        diags = [np.diag(arr, k=i) for i in range(-arr.shape[0] + 1, arr.shape[1])]
        flipped_diags = [np.diag(np.fliplr(arr), k=i) for i in range(-arr.shape[0] + 1, arr.shape[1])]
        diags.extend(flipped_diags)
        for diag in diags:
            # if diag.size < self.num_to_win:
            #     continue
            results.append(func(diag.reshape(-1)))

        return results

    def extract_all_vectors(self, board: np.ndarray):
        vectors = board.tolist() + board.T.tolist()
        vectors.extend([np.diag(board, k=i) for i in range(-board.shape[0] + 1, board.shape[1])])
        vectors.extend([np.diag(np.fliplr(board), k=i) for i in range(-board.shape[0] + 1, board.shape[1])])
        # pad vectors with -3's to have the same length as the board size and stack them
        vectors = [np.pad(vector, (0, self.board_size - len(vector)), constant_values=-3) for vector in vectors]
        return np.array(vectors)

    def check_win(self, player: int, board=None) -> bool:
        # if self.num_to_win == self.board_size:
        #     return self.check_full_win(player, board=board)
        # else:
        return self.check_partial_win(player, self.num_to_win, board=board)

    def check_full_win(self, player: int, board=None) -> bool:
        """
        This function checks if the supplied player has won the game with a full win (num_to_win == board_size).
        :param player: The player to check for (1 or -1).
        :param board: The board to check on. If None, the current board is used.
        :return: True if the player has won, False otherwise.
        """
        if board is None:
            board = self.get_board()
        matches = self.full_iterate_array(board, lambda part: np.all(part == player))
        for match in matches:
            if not isinstance(match, str) and match:
                return True

        return False

    def check_partial_win(self, player: int, n: int, board=None) -> bool:
        """
        This function checks if the supplied player has won the game with a partial win (num_to_win < board_size).

        :param player: The player to check for (1 or -1).
        :param n: The number of consecutive pieces needed to win.
        :param board: The board to check on. If None, the current board is used.
        :return: True if the player has won, False otherwise.
        """
        if board is None:
            board = self.get_board()
        matches = self.full_iterate_array_all_diags(board,
                                                    lambda part:
                                                    np.convolve((part == player), np.ones(n, dtype=np.int8),
                                                                "valid"))

        for match in matches:
            if np.any(match == n):
                return True

        return False

    def check_partial_win_vectorized(self, player: int, n: int, board=None) -> bool:
        if board is None:
            board = self.get_board()
        vectors = th.tensor(self.extract_all_vectors(board)).unsqueeze(0)
        weight = th.ones((1, 1, 1, n), dtype=th.long)
        vectors_where_player = th.where(vectors == player, 1, 0).long()
        res = th.nn.functional.conv2d(vectors_where_player, weight=weight)
        return th.any(res == n).item()

    def check_partial_win_to_index(self, player: int, n: int, board=None) -> dict[tuple, str] | dict:
        """
        This variation of check_partial_win returns the index of the first partial win found.
        The index being the index of the first piece in the winning sequence.
        :param player: The player to check for (1 or -1).
        :param n: The number of consecutive pieces needed to win.
        :param board: The board to check on. If None, the current board is used.
        :return: A dictionary containing the index and the position of the winning sequence.
        """
        if board is None:
            board = self.get_board()
        indices = self.full_iterate_array(board,
                                          lambda part:
                                          np.convolve((part == player), np.ones(n, dtype=int),
                                                      "valid"))
        indices = [x.tolist() if not isinstance(x, str) else x for x in indices]

        pos = "row"
        for vector in indices:
            if isinstance(vector, str):
                pos = vector
                continue
            for el in vector:
                if el == n:
                    vector_index = indices.index(vector)
                    pos_index = indices.index(pos)
                    # num_strings_before = len([x for index,x in enumerate(indices) if isinstance(x,str) and index < pos_index])
                    element_index = vector_index - (pos_index + 1) if vector_index > pos_index else indices.index(
                        vector, pos_index)
                    diag_idx = {
                        "index": ((self.board_size - 1) - element_index, 0)} if element_index < self.board_size else {
                        "index": (0, element_index - self.board_size)}
                    diag_idx["pos"] = pos
                    match pos:
                        case "row":
                            return {"index": (element_index, 0), "pos": pos}
                        case "col":
                            return {"index": (0, element_index), "pos": pos}
                        case "diag_left":
                            return diag_idx
                        case "diag_right":
                            return diag_idx

                        # case "diag_right":

        return {"index": None, "pos": None}

    def return_index_if_valid(self, index: tuple, return_on_fail: tuple = ()) -> tuple:
        if index[0] < 0 or index[0] >= self.board_size or index[1] < 0 or index[1] >= self.board_size:
            return return_on_fail
        return index

    def make_fresh_instance(self):
        return TicTacToeGameManager(self.board_size, self.headless, num_to_win=self.num_to_win)

    def get_previous(self, index: tuple, pos: str, n: int):
        if np.array(index).all() == 0:
            return index
        match pos:
            case "row":
                return self.return_index_if_valid((index[0], index[1] - n), return_on_fail=index)
            case "col":
                return self.return_index_if_valid((index[0] - n, index[1]), return_on_fail=index)
            case "diag_left":
                return self.return_index_if_valid((index[0] - n, index[1] - n), return_on_fail=index)
            case "diag_right":
                return self.return_index_if_valid((index[0] - n, index[1] + n), return_on_fail=index)

    def get_next(self, index: tuple, pos: str, n: int):
        if np.array(index).all() == self.board_size - 1:
            return index
        match pos:
            case "row":
                return self.return_index_if_valid((index[0], index[1] + n), return_on_fail=index)
            case "col":
                return self.return_index_if_valid((index[0] + n, index[1]), return_on_fail=index)
            case "diag_left":
                return self.return_index_if_valid((index[0] + n, index[1] + n), return_on_fail=index)
            case "diag_right":
                return self.return_index_if_valid((index[0] + n, index[1] - n), return_on_fail=index)

    def is_board_full(self, board=None) -> bool:
        if board is None:
            board = self.get_board()
        return np.all(board != 0)

    def get_board(self):
        return self.board.copy()

    def reset(self):
        self.board = self.initialize_board()
        return self.board.copy()

    def get_board_size(self):
        return self.board_size

    def render(self) -> bool:
        if self.headless:
            return False

        self.screen.fill((0, 0, 0))
        self._draw_board()
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == self.player:
                    self._draw_circle(col * 100 + 50, row * 100 + 50)

                elif self.board[row][col] == self.enemy_player:
                    self._draw_cross(col * 100 + 50, row * 100 + 50)

        pg.event.pump()
        pg.display.flip()
        return True

    def _draw_circle(self, x, y) -> None:
        if self.headless:
            return
        pg.draw.circle(self.screen, "green", (x, y), 40, 1)

    def _draw_cross(self, x, y) -> None:
        if self.headless:
            return
        pg.draw.line(self.screen, "red", (x - 40, y - 40), (x + 40, y + 40), 1)
        pg.draw.line(self.screen, "red", (x + 40, y - 40), (x - 40, y + 40), 1)

    def _draw_board(self):
        for x in range(0, self.board_size * 100, 100):
            for y in range(0, self.board_size * 100, 100):
                pg.draw.rect(self.screen, (255, 255, 255), pg.Rect(x, y, 100, 100), 1)

    def is_empty(self, index: tuple) -> bool:
        return self.board[index] == 0

    def get_valid_moves(self, observation: np.ndarray, player: int or None = None) -> list:
        """
        Legal moves are the empty spaces on the board.
        :param observation: A 2D numpy array representing the current state of the game.
        :param player: The player to check for. Since the game is symmetric, this is ignored.
        :return: A list of legal moves.
        """
        legal_moves = []
        observation = observation.reshape(self.board_size, self.board_size)
        for row in range(self.board_size):
            for col in range(self.board_size):
                if observation[row][col] == 0:
                    legal_moves.append([row, col])
        return legal_moves

    def pygame_quit(self) -> bool:
        if self.headless:
            return False
        pg.quit()
        return True

    def get_click_coords(self):
        if self.headless:
            return
        mouse_pos = (x // 100 for x in pg.mouse.get_pos())
        if pg.mouse.get_pressed()[0]:  # left click
            return mouse_pos

    def get_human_input(self, board: np.ndarray):
        if self.headless:
            return
        while True:
            self.check_pg_events()
            if self.get_click_coords() is not None:
                x, y = self.get_click_coords()
                if board[y][x] == 0:
                    return y, x

            # time.sleep(1 / 60)

    def check_pg_events(self):
        if self.headless:
            return
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.pygame_quit()
                sys.exit(0)

    def network_to_board(self, move):
        """
        Converts an integer move from the network to a board index.
        :param move: An integer move selected from the network probabilities.
        :return: A tuple representing the board index (int,int).
        """
        return np.unravel_index(move, self.board.shape)

    def save_screenshot_with_probabilities(self, action_probs, path):
        if self.headless:
            return
        plt.figure(figsize=(15, 10))
        labels, probabilities = zip(*action_probs.items())
        print(probabilities)
        labels = [f"{np.unravel_index(x, self.board.shape)[0]};{np.unravel_index(x, self.board.shape)[1]}" for x in
                  labels]
        sns.barplot(x=labels, y=probabilities)
        plt.xticks(rotation=90)
        plt.xlabel("Move")
        plt.ylabel("Probability")
        plt.title("Action probabilities")
        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches="tight")
        buf.seek(0)
        plot_img = Image.open(buf)

        # Convert the pg surface to an image
        surface_buffer = pg.image.tostring(self.screen, 'RGBA')
        surface_img = Image.frombytes('RGBA', self.screen.get_size(), surface_buffer)

        # Concatenate the images vertically
        total_height = plot_img.height + surface_img.height
        combined_img = Image.new('RGB', (max(plot_img.width, surface_img.width), total_height))
        combined_img.paste(plot_img, (0, 0))
        combined_img.paste(surface_img, (0, plot_img.height))
        combined_img.save(path)

    @staticmethod
    def get_canonical_form(board, player) -> np.ndarray:
        return board * player

    def get_next_state(self, board: np.ndarray, action: int or tuple, player: int) -> np.ndarray:
        if isinstance(action, int):
            action = self.network_to_board(action)
        board_ = board.copy()
        board_[action] = player
        return board_

    def set_headless(self, val: bool):
        self.headless = val

    def __str__(self):
        return str(self.board).replace('1', 'X').replace('-1', 'O')
#
# if __name__ == "__main__":
#     sample_arr = np.array([[-1,1,-1,1,1],[-1,1,1,-1,-1],[1,-1,1,1,0],[-1,1,1,-1,0],[0,0,-1,1,1]])
#     game_manager = GameManager(5, True,num_to_win=3)
# res = game_manager.check_partial_win_vectorized(1,3,sample_arr)
# res = game_manager.game_result(-1,sample_arr)
# print(res)
# print(game_manager.check_partial_win(1,3,sample_arr))
