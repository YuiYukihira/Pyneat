import pytest
import pyneat
import random
from math import exp

def test_graph():
    graph = pyneat.Graph({
        'e': {'nodes': [('d', 1, 0)], 'type': 'exit'},
        'd': {'nodes': [], 'type': 'enter'},
        'c': {'nodes': [('a', 1, 0), ('b', 1, 0)], 'type': 'exit'},
        'b': {'nodes': [], 'type': 'enter'},
        'a': {'nodes': [], 'type': 'enter'}
    })
    a = graph.run({'a': 1, 'b': 1, 'd': 1})
    print(a)
    assert a[0][0] > 0.731
    assert a[0][0] < 0.732
    assert a[0][1] > 1.462
    assert a[0][1] < 1.463
    #assert graph.run({'a': 1, 'b': 1, 'd': 1}) == ([1/(1+exp(-1)), 1/(1+exp(-2))], ['e', 'c'])

def test_graph_disconnected():
    graph = pyneat.Graph({
        'd': {'nodes': [], 'type': 'exit'},
        'c': {'nodes': [('b', 1, 0), ('a', 1, 0)], 'type': 'hidden'},
        'b': {'nodes': [], 'type': 'enter'},
        'a': {'nodes': [], 'type': 'enter'}
    })
    assert graph.run({'a': 1, 'b': 1}) == ([0.0], ['d'])

def test_neat_graph_convert():
    graph = pyneat.Graph({
        'e': {'nodes': [('d', 1, 0)], 'type': 'exit'},
        'd': {'nodes': [], 'type': 'enter'},
        'c': {'nodes': [('a', 1, 0), ('b', 1, 0)], 'type': 'exit'},
        'b': {'nodes': [], 'type': 'enter'},
        'a': {'nodes': [], 'type': 'enter'}
    })
    ngraph = pyneat.NeatGraph(pyneat.Genotype(
        nodes=[
            pyneat.NodeGene('e', 'exit'),
            pyneat.NodeGene('d', 'enter'),
            pyneat.NodeGene('c', 'exit'),
            pyneat.NodeGene('b', 'enter'),
            pyneat.NodeGene('a', 'enter')
        ],
        conns={
            1: pyneat.ConnGene('a', 'c', 1.0, True),
            2: pyneat.ConnGene('b', 'c', 1.0, True),
            3: pyneat.ConnGene('d', 'e', 1.0, True)
        }, mutation_rate=100))
    assert ngraph.phenotype.shape == graph.shape

def test_neat_graph_run():
    ngraph = pyneat.NeatGraph(pyneat.Genotype(
        nodes=[
            pyneat.NodeGene('e', 'exit'),
            pyneat.NodeGene('d', 'enter'),
            pyneat.NodeGene('c', 'exit'),
            pyneat.NodeGene('b', 'enter'),
            pyneat.NodeGene('a', 'enter')
        ],
        conns={
            1: pyneat.ConnGene('a', 'c', 1.0, True),
            2: pyneat.ConnGene('b', 'c', 1.0, True),
            3: pyneat.ConnGene('d', 'e', 1.0, True)
        }, mutation_rate=100))
    a = ngraph.run({'a': 1, 'b': 1, 'd': 1})
    assert a[0][0] > 0.73
    assert a[0][0] < 0.74
    assert a[0][1] > 1.46
    assert a[0][1] < 1.47
    #assert ngraph.run({'a': 1, 'b': 1, 'd': 1}) == ([0.7310586, 1.4621172], ['e', 'c'])

def test_graph_empty():
    graph = pyneat.Graph({})
    assert graph.run({}) is None
    assert graph.run({'a': 1}) is None

def test_neat_graph_empty():
    graph = pyneat.NeatGraph(pyneat.Genotype([], {}, 100))
    assert graph.genotype.nodes == []
    assert graph.genotype.conns == {}
    assert graph.phenotype.shape == {}

def test_neat_controller_init():
    controller = pyneat.Controller(10, 10, {'a': 'enter', 'b': 'enter', 'c': 'exit'})
    for i in range(0, 10):
        for j in range(0, 10):
            assert isinstance(controller.graphs[i][j], pyneat.NeatGraph)
            nodes = [node.id for node in controller.graphs[i][j].genotype.nodes]
            assert nodes == ['a', 'b', 'c']
            assert controller.graphs[i][j].genotype.conns == {}
            assert controller.scores[i][j] == 0

def test_neat_controller_scoring():
    controller = pyneat.Controller(2,2, {'a': 'enter', 'b': 'enter', 'c': 'exit'})
    assert controller.scores == {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}}
    controller.game_over(10)
    assert controller.scores == {0: {0: 10, 1: 0}, 1: {0: 0, 1: 0}}
    controller.game_over(555)
    assert controller.scores == {0: {0: 10, 1: 555}, 1: {0: 0, 1: 0}}
    controller.game_over(1)
    assert controller.scores == {0: {0: 10, 1: 555}, 1: {0: 1, 1: 0}}

def test_neat_controller_breeding():
    random.seed(1)
    controller = pyneat.Controller(1,5,{'a': 'enter', 'b': 'enter', 'c': 'exit'})
    controller.game_over(1)
    controller.game_over(2)
    controller.game_over(3)
    controller.game_over(4)
    controller.game_over(5)
    assert len(controller.graphs[0]) == 5
    assert controller.graphs[0] != {}
    for graph_g in controller.graphs[0].values():
        node_ids = {i.id for i in graph_g.genotype.nodes}
        assert set(controller.required_nodes.keys()) == {'a', 'b', 'c'}
        assert node_ids != set()
        for node_id in node_ids:
            assert node_id in {'a', 'b', 'c'}
    controller.game_over(1)
    controller.game_over(2)
    controller.game_over(3)
    controller.game_over(4)
    controller.game_over(5)
    assert len(controller.graphs[0]) == 5
    assert controller.graphs[0] != {}
    for graph_g in controller.graphs[0].values():
        node_ids = {i.id for i in graph_g.genotype.nodes}
        assert set(controller.required_nodes.keys()) == {'a', 'b', 'c'}
        assert node_ids != set()
        for node_id in node_ids:
            assert node_id in {'a', 'b', 'c'}

if __name__ == '__main__':
    test_graph()
