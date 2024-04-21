import base64

import numpy as np
from fastapi import APIRouter,FastAPI

from mu_alpha_zero.General.mz_game import MuZeroGame


class GymEnvWrapper:
    def __init__(self, env: MuZeroGame):
        self.env_instance = env
        self.router = APIRouter()
        self.register_routes()
        self.app = FastAPI()
        self.app.include_router(self.router)

    def register_routes(self):
        self.router.add_api_route("/frame_skip_step/{action}/{player}/{frame_skip}", self.frame_skip_step,
                                  methods=["GET"])
        self.router.add_api_route("/reset", self.reset, methods=["GET"])
        self.router.add_api_route("/get_noop", self.get_noop, methods=["GET"])
        self.router.add_api_route("/get_num_actions", self.get_num_actions, methods=["GET"])
        self.router.add_api_route("/game_result/{player}", self.game_result, methods=["GET"])

    def frame_skip_step(self, action: int, player: int, frame_skip: int):
        if player == 0:
            player = None
        state, reward, done = self.env_instance.frame_skip_step(action, player, frame_skip)
        return {"state": self.encode_array(state), "reward": reward, "done": done,"shape":self.encode_shape(state)}

    def reset(self):
        state = self.env_instance.reset()
        return {"state": self.encode_array(state),"shape":self.encode_shape(state)}

    def get_noop(self):
        return self.env_instance.get_noop()

    def get_num_actions(self):
        return int(self.env_instance.get_num_actions())

    def game_result(self, player: int):
        if player == 0:
            player = None
        return self.env_instance.game_result(player)

    def encode_array(self, array: np.ndarray):
        return base64.b64encode(array.tobytes()).decode("utf-8")

    def encode_shape(self, array: np.ndarray):
        return ",".join([str(x) for x in array.shape])
