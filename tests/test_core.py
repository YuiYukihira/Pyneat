import pytest
import pyneat
import random
from math import exp
from os import remove, getpid
import time
import psutil

def type_dec(error, message):
    def type_dec_inner(func):
        def func_wrapper():
            with pytest.raises(error, message=message):
                func()
        return func_wrapper
    return type_dec_inner

def test_type_node():
    @type_dec(TypeError, "Expecting TypeError")
    def test1():
        pyneat.NodeGene(1, 'enter')
    @type_dec(TypeError, "Expecting TypeError")
    def test2():
        pyneat.NodeGene("1", 1)
    @type_dec(ValueError, "Expecting ValueError")
    def test3():
        pyneat.NodeGene("1", "1")
    test1()
    test2()
    test3()

def test_type_conn():
    @type_dec(TypeError, "Expecting TypeError")
    def test1():
        pyneat.ConnGene(1, "1", 0.1, True)
    @type_dec(TypeError, "Expecting TypeError")
    def test2():
        pyneat.ConnGene("1", 1, 0.1, True)
    @type_dec(TypeError, "Expecting TypeError")
    def test3():
        pyneat.ConnGene("1", "1", "0.1", True)
    @type_dec(TypeError, "Expecting TypeError")
    def test4():
        pyneat.ConnGene("1", "1", 0.1, "True")
    test1()
    test2()
    test3()
    test4()

def test_type_genotype():
    @type_dec(TypeError, "Expecting TypeError")
    def test1():
        pyneat.Genotype('w', {}, 1)
    @type_dec(TypeError, "Expecting TypeError")
    def test2():
        pyneat.Genotype([], 'w', 1)
    @type_dec(TypeError, "Expecting TypeError")
    def test3():
        pyneat.Genotype(["w"], {}, 1)
    @type_dec(TypeError, "Expecting TypeError")
    def test4():
        pyneat.Genotype([], {"1": pyneat.ConnGene("1", "2", 1.0, True)}, 1)
    @type_dec(TypeError, "Expecting TypeError")
    def test5():
        pyneat.Genotype([], {1: "w"}, 1)
    @type_dec(TypeError, "Expecting TypeError")
    def test6():
        pyneat.Genotype([], {}, "1")
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()

def test_type_controller():
    @type_dec(TypeError, "Expecting TypeError")
    def test1():
        pyneat.Controller("1",1,{"a": "enter"})
    @type_dec(TypeError, "Expecting TypeError")
    def test2():
        pyneat.Controller(1,"1",{"a": "enter"})
    @type_dec(TypeError, "Expecting TypeError")
    def test3():
        pyneat.Controller(1,1,{1: "enter"})
    @type_dec(TypeError, "Expecting TypeError")
    def test4():
        pyneat.Controller(1,1,{"a": 1})
    @type_dec(ValueError, "Expecting ValueError")
    def test5():
        pyneat.Controller(1,1,{"a": "a"})
    test1()
    test2()
    test3()
    test4()
    test5()







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

def test_breed_repopulation():
    controller = pyneat.Controller(5,5,{'a':'enter','b':'enter','c':'exit'})
    assert controller.genera_count==5 and controller.species_count==5
    assert len(controller.graphs)==5
    for genera in controller.graphs.values():
        assert len(genera)==5
        for i in range(0, 25):
            controller.game_over(i)
    assert controller.genera_count==5 and controller.species_count==5
    assert len(controller.graphs)==5
    for genera in controller.graphs.values():
        assert len(genera)==5
    for i in range(0, 25):
        controller.game_over(i)
    assert controller.genera_count==5 and controller.species_count==5
    assert len(controller.graphs)==5
    for genera in controller.graphs.values():
        assert len(genera)==5

def test_save_load():
    c1 = pyneat.Controller(50, 50, {'a': 'enter', 'b': 'enter', 'c': 'exit'})
    c1.save_state("NEATsave.pkl")
    c2 = pyneat.Controller.load_state("NEATsave.pkl")
    assert c1.genera_count == c2.genera_count
    assert c1.species_count == c1.species_count
    assert {i: {ji: jk.genotype for ji, jk in j.items()} for i, j in c1.graphs.items()} ==     {i: {ji: jk.genotype for ji, jk in j.items()} for i, j in c2.graphs.items()}
    assert c1.scores == c2.scores
    assert c1.current == c2.current
    assert c1.required_nodes == c2.required_nodes
    assert c1.innovation_dict.copy() == c2.innovation_dict.copy()
    remove("NEATsave.pkl")


def test_controller_time_performance():
    times = []
    for i in range(3):
        start = time.perf_counter()
        controller = pyneat.Controller(50,50,{'a':'enter','b':'enter','c': 'exit'})
        end = time.perf_counter()
        print(f'time taken: {end-start}')
        times.append(end-start)
    assert sum(times)/len(times) < 8

def test_controller_memory_performance():
    start_memory = []
    end_memory = []
    process = psutil.Process(getpid())
    for i in range(3):
        controller = pyneat.Controller(5,5,{'a':'enter','b':'enter','c':'exit'})
        start_memory.append(process.memory_info().rss)
        for i in range(0, 25):
            controller.game_over(i)
        end_memory.append(process.memory_info().rss)
    avg_start = sum(start_memory)/len(start_memory)
    avg_end = sum(end_memory)/len(end_memory)
    assert avg_end <= avg_start
