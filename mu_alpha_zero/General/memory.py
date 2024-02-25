from abc import ABC, abstractmethod


class GeneralMemoryBuffer(ABC):
    @abstractmethod
    def add(self, experience):
        """
        Add a single experience to the buffer.
        """
        pass

    @abstractmethod
    def add_list(self, experience_list):
        """
        Add a list of experiences to the buffer.
        """
        pass

    @abstractmethod
    def batch(self, batch_size):
        """
        Return a batch of experiences.
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        Return the length of the buffer.
        """
        pass

    @abstractmethod
    def make_fresh_instance(self):
        pass
