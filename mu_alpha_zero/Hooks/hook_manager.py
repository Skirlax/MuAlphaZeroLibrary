from mu_alpha_zero.Hooks.hook_callables import HookCallable
from mu_alpha_zero.Hooks.hook_point import HookPoint, HookAt


class HookManager:
    def __init__(self):
        self.hooks = {}

    def register_method_hook(self, where: HookPoint, hook: HookCallable):
        self.hooks[where] = hook

    def process_hook_executes(self, cls: object, fn_name: str, file: str, at: HookAt, args: tuple = ()):
        file_name = file.replace("\\", "/").split("/")[-1]
        for hook_point, hook_callable in self.hooks.items():
            if hook_point.here(file_name, fn_name, at):
                hook_callable.execute(cls, *args)
                return
