import parsl
from parsl.app.app import python_app, bash_app
parsl.load()


class Parallel:

    def __init__(self):
        self.__operations = []
        self.__operation_futures = []

    def add_operation(self, callback, *args):
        self.__operations.append({callback, *args})

    def add_operation_complete_listener(self, callback):
        for operation_future in self.__operation_futures:
            result = operation_future.result()
            callback(result)

    @python_app
    def execute(callback, *args):
        return callback(args)

    def execute_all(self):
        for operation in self.__operations:
            self.__operation_futures.append(
                self.execute(operation.callback, operation.args))
