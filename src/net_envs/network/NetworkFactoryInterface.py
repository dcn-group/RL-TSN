import abc


class NetworkFactoryInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def product(self, graph, flows, *args, **kwargs):
        pass
