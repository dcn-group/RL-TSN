from Strategy.topo_strategy.BarabasiAlbertStrategy import BarabasiAlbertStrategy
from Strategy.topo_strategy.ErdosRenyiStrategy import ErdosRenyiStrategy
from Strategy.topo_strategy.RandomRegularGraphStrategy import RandomRegularGraphStrategy
from Strategy.topo_strategy.TopoStrategy import TopoStrategy
from Strategy.topo_strategy import WattsStrogatzStrategy
from RL_env.type import TOPO_STRATEGY


class TopoStrategyFactory(object):

    @staticmethod
    def get_instance(*args, **kwargs) -> TopoStrategy:
        if kwargs['strategy'] == TOPO_STRATEGY.ER_STRATEGY:
            return ErdosRenyiStrategy(t=kwargs['type'], n=kwargs['n'], m=kwargs['m'], p=kwargs['p'])
        elif kwargs['strategy'] == TOPO_STRATEGY.BA_STRATEGY:
            return BarabasiAlbertStrategy(n=kwargs['n'], m=kwargs['m'])
        elif kwargs['strategy'] == TOPO_STRATEGY.RRG_STRATEGY:
            return RandomRegularGraphStrategy(d=kwargs['d'], n=kwargs['n'])
        elif kwargs['strategy'] == TOPO_STRATEGY.WS_STRATEGY:
            return WattsStrogatzStrategy(n=kwargs['n'], k=kwargs['k'], p=kwargs['p'])
        else:
            raise RuntimeError("allocating strategy doesn't exist")
