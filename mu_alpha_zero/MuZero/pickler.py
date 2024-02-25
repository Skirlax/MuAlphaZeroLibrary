import os
import pickle

from portalocker import Lock


class DataPickler:
    """
    Class for efficiently storing and retrieving data from the file system.
    """

    def __init__(self, pickle_dir: str):
        self.processed_count = 0
        self.pickle_dir = self.__init_pickle_dir(pickle_dir)

    def __init_pickle_dir(self, pickle_dir: str) -> str:
        os.makedirs(pickle_dir, exist_ok=True)
        return pickle_dir

    def pickle_buffer(self, buffer: list):
        for index in range(len(buffer[0])):
            data = [item[index] for item in buffer]
            with Lock(f"{self.pickle_dir}/item_{index}.pkl", "ab", timeout=90) as f:
                pickle.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
        self.processed_count += 1

    def load_index(self, index: int):
        file = f"{self.pickle_dir}/item_{index}.pkl"
        data = []
        with Lock(file, "rb", timeout=90) as f:
            while True:
                try:
                    data.extend(pickle.load(f))
                except EOFError:
                    break

        return data

    def load_all(self, batch_size: int, indexes: list, K: int = 1):
        files = [x for x in os.listdir(self.pickle_dir) if x.endswith(".pkl")]
        files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        data = [[] for _ in range(len(files))]
        for i, file in enumerate(files):
            with Lock(f"{self.pickle_dir}/{file}", "rb", timeout=90) as f:
                last_len = 0
                while True:
                    if len(data[i]) >= batch_size:
                        break
                    try:
                        file_data = pickle.load(f)
                        for index in indexes:
                            if last_len + len(file_data) > index >= last_len:
                                if last_len + len(file_data) - K < index:
                                    index -= K - ((last_len + len(file_data)) - index)
                                index -= last_len
                                data_point = file_data[index:index + K]
                                data[i].extend(data_point)

                        last_len += len(file_data)
                    except EOFError:
                        break
        return [(x[0], x[1], (x[2][0], x[2][1], x[2][2]), x[3]) for x in zip(*data)]

    def clear_dir(self):
        files = [x for x in os.listdir(self.pickle_dir) if x.endswith(".pkl")]
        for file in files:
            os.remove(f"{self.pickle_dir}/{file}")
        self.processed_count = 0
