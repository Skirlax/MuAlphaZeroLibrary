from abc import abstractmethod, ABC
from typing import Callable


class HookCallable(ABC):

    @abstractmethod
    def execute(self, cls: object, *args):
        """
        Execute the hook
        :return: The return of the hook
        """
        pass


class HookMethodCallable(HookCallable):
    def __init__(self, method: Callable, args: tuple):
        super().__init__()
        self.__method = method
        self.__args = args

    def execute(self, cls: object, *args):
        return self.__method(cls, *self.__args, *args)



