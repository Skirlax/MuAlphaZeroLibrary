import enum


class HookAt(enum.Enum):
    HEAD = "head"
    TAIL = "tail"
    MIDDLE = "middle"
    ALL = "all"


class HookPoint:
    def __init__(self, at: HookAt, file: str, fn_name: str):
        self.__at = at
        self.__file = file
        self.__function_name = fn_name

    @property
    def at(self) -> HookAt:
        return self.__at

    @property
    def file(self) -> str:
        return self.__file

    @property
    def function_name(self) -> str:
        return self.__function_name

    def here(self, file: str, function_name: str, at: HookAt):
        return file == self.file and function_name == self.function_name and (at == self.at or at == HookAt.ALL)
