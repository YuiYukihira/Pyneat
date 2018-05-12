import pytest
import pyneat
import random
from math import exp
from os import remove, getpid
import time
import psutil
import logging
import copy

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
        pyneat.Controller("1", 1, {"a": "enter"})
    @type_dec(TypeError, "Expecting TypeError")
    def test2():
        pyneat.Controller(1, "1", {"a": "enter"})
    @type_dec(TypeError, "Expecting TypeError")
    def test3():
        pyneat.Controller(1, 1, {1: "enter"})
    @type_dec(TypeError, "Expecting TypeError")
    def test4():
        pyneat.Controller(1, 1, {"a": 1})
    @type_dec(ValueError, "Expecting ValueError")
    def test5():
        pyneat.Controller(1, 1, {"a": "a"})
    test1()
    test2()
    test3()
    test4()
    test5()

def test_graph_output():
    shape = {
        'd': {'nodes': [('c', 1.0, 0.0)], 'type': 'exit'},
        'e': {'nodes': [('b', 1.0, 0.0)], 'type': 'exit'},
        'c': {'nodes': [('a', 1.0, 0.0), ('b', 1.0, 0.0)], 'type': 'hidden'},
        'b': {'nodes': [], 'type': 'enter'},
        'a': {'nodes': [], 'type': 'enter'},
    }
    graph = pyneat.Graph(shape)
    out = graph.run({'a': 1.0, 'b': 1.0})
    for i, j in zip(out[1], out[0]):
        assert (i == 'e' and round(float(j), 7) == 0.6750376) or (i == 'd' and round(float(j), 7) == 0.7941419)

    shape = {
        'd': {'nodes': [], 'type': 'exit'},
        'c': {'nodes': [('a', 1.0, 0.0), ('b', 1.0, 0.0)], 'type': 'hidden'},
        'b': {'nodes': [], 'type': 'enter'},
        'a': {'nodes': [], 'type': 'enter'}
    }
    graph = pyneat.Graph(shape)
    out = graph.run({'a': 1.0, 'b': 1.0})
    assert out == ([0.0], ['d'])

    graph = pyneat.Graph({})
    assert graph.run({}) is None
    assert graph.run({'a': 1.0}) is None

def test_neat_graph_output():
    genotype = pyneat.Genotype(
        [
            pyneat.NodeGene('d', 'exit'),
            pyneat.NodeGene('e', 'exit'),
            pyneat.NodeGene('c', 'hidden'),
            pyneat.NodeGene('b', 'enter'),
            pyneat.NodeGene('a', 'enter')],
        {
            1: pyneat.ConnGene('c', 'd', 1.0, True),
            2: pyneat.ConnGene('b', 'e', 1.0, True),
            3: pyneat.ConnGene('a', 'c', 1.0, True),
            4: pyneat.ConnGene('b', 'c', 1.0, True)}, 100)
    graph = pyneat.NeatGraph(genotype)
    out = graph.run({'a': 1.0, 'b': 1.0})
    for i, j in zip(out[1], out[0]):
        assert (i == 'e' and round(float(j), 7) == 0.6750376) or (i == 'd' and round(float(j), 7) == 0.7941419)

    genotype = pyneat.Genotype(
        [
            pyneat.NodeGene('d', 'exit'),
            pyneat.NodeGene('c', 'hidden'),
            pyneat.NodeGene('b', 'enter'),
            pyneat.NodeGene('a', 'enter')],
        {
            3: pyneat.ConnGene('a', 'c', 1.0, True),
            4: pyneat.ConnGene('b', 'c', 1.0, True)}, 100)
    graph = pyneat.NeatGraph(genotype)
    out = graph.run({'a': 1.0, 'b': 1.0})
    assert out == ([0.0], ['d'])

    graph = pyneat.NeatGraph(pyneat.Genotype([], {}, 100))
    assert graph.run({'a': 1.0}) is None
    assert graph.run({}) is None

def test_mutations():
    breed_controller = pyneat.core.BreedController(
        {'a': 'enter', 'b': 'enter', 'c': 'exit'},
        {0: {0: 0}},
        {0: {0: None}},
        1,
        1)
    breed_controller.innovation_dict = {1: {'in': 'a', 'out': 'c'}}
    base_genotype = pyneat.Genotype(
        [
            pyneat.NodeGene('a', 'enter'),
            pyneat.NodeGene('b', 'enter'),
            pyneat.NodeGene('c', 'exit')],
        {1: pyneat.ConnGene('a', 'c', 1.0, True)},
        100)

    # mutation type 0:
    random.seed(1)
    geno_1 = breed_controller.mutate(copy.deepcopy(base_genotype), 0)
    cgeno_1 = pyneat.Genotype(
        [
            pyneat.NodeGene('a', 'enter'),
            pyneat.NodeGene('b', 'enter'),
            pyneat.NodeGene('c', 'exit')],
        {1: pyneat.ConnGene('a', 'c', 1.0, True)},
        17)
    assert geno_1 == cgeno_1

    # mutation type 1:
    random.seed(1)
    geno_2 = breed_controller.mutate(copy.deepcopy(base_genotype), 1)
    cgeno_2 = pyneat.Genotype(
        [
            pyneat.NodeGene('a', 'enter'),
            pyneat.NodeGene('b', 'enter'),
            pyneat.NodeGene('c', 'exit'),
            pyneat.NodeGene('c386bbc4-cd61-3e30-d8f1-6adf91b7584a', 'hidden')
        ],
        {1: pyneat.ConnGene('a', 'c', 1.0, False),
         2: pyneat.ConnGene('a', 'c386bbc4-cd61-3e30-d8f1-6adf91b7584a', 1.0, True),
         3: pyneat.ConnGene('c386bbc4-cd61-3e30-d8f1-6adf91b7584a', 'c', 1.0, True)
        },
        100)
    assert geno_2 == cgeno_2

    # mutation type 2:
    random.seed(1)
    geno_3 = breed_controller.mutate(copy.deepcopy(base_genotype), 2)
    assert geno_3.nodes == base_genotype.nodes
    assert geno_3.mutation_rate == base_genotype.mutation_rate
    assert len(geno_3.conns) == 2
    assert list(geno_3.conns.keys()) == [1, 4]
    assert geno_3.conns[1] == pyneat.ConnGene('a', 'c', 1.0, True)
    t_gene = pyneat.ConnGene('b', 'c', 0.9908702, True)
    assert geno_3.conns[4].in_node == t_gene.in_node
    assert geno_3.conns[4].out_node == t_gene.out_node
    assert round(float(geno_3.conns[4].weight), 7) == t_gene.weight
    assert geno_3.conns[4].enabled == t_gene.enabled

    # mutation type 3:
    random.seed(1)
    geno_4 = breed_controller.mutate(copy.deepcopy(base_genotype), 3)
    cgeno_4 = pyneat.Genotype(
        [
            pyneat.NodeGene('a', 'enter'),
            pyneat.NodeGene('b', 'enter'),
            pyneat.NodeGene('c', 'exit')],
        {1: pyneat.ConnGene('a', 'c', 1.0, False)},
        100)
    assert geno_4 == cgeno_4

    # mutation type 4:
    random.seed(1)
    geno_5 = breed_controller.mutate(copy.deepcopy(base_genotype), 4)
    cgeno_5 = pyneat.Genotype(
        [
            pyneat.NodeGene('a', 'enter'),
            pyneat.NodeGene('b', 'enter'),
            pyneat.NodeGene('c', 'exit')],
        {1: pyneat.ConnGene('a', 'c', 0.5101381, True)},
        100)
    assert geno_5.nodes == cgeno_5.nodes
    assert geno_5.mutation_rate == cgeno_5.mutation_rate
    assert len(geno_5.conns) == 1
    assert list(geno_5.conns.keys()) == [1]
    assert geno_5.conns[1].in_node == cgeno_5.conns[1].in_node
    assert geno_5.conns[1].out_node == cgeno_5.conns[1].out_node
    assert round(float(geno_5.conns[1].weight), 7) == cgeno_5.conns[1].weight
    assert geno_5.conns[1].enabled == cgeno_5.conns[1].enabled

def test_breed_genotypes():
    random.seed(1)
    bc = pyneat.core.BreedController(
        {'a': 'enter', 'b': 'enter', 'c': 'exit'},
        {0: {0: None, 1: None}},
        {0: {0: 0, 1: 1}},
        2,
        1)
    geno1 = pyneat.Genotype(
        [pyneat.NodeGene('a', 'enter'),
         pyneat.NodeGene('b', 'enter'),
         pyneat.NodeGene('c', 'exit')],
        {}, 43)
    geno2 = pyneat.Genotype(
        [pyneat.NodeGene('a', 'enter'),
         pyneat.NodeGene('b', 'enter'),
         pyneat.NodeGene('c', 'exit')],
        {1: pyneat.ConnGene('a', 'c', 1.0, True)}, 76)
    c_geno = bc.create_new_offspring((geno1, 1), (geno2, 0), {1: {'in': 'a', 'out': 'c'}})
    for i in c_geno.nodes:
        print(i)
    for i, j in c_geno.conns.items():
        print(f'{i}: {j}')
    assert c_geno.nodes == geno2.nodes
    assert c_geno.conns == geno2.conns
    assert c_geno.mutation_rate == geno1.mutation_rate

def test_controller_game_over():
    con = pyneat.Controller(5, 5, {'a': 'enter', 'b': 'enter', 'c': 'exit'})
    assert con.scores == {0: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                          1: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                          2: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                          3: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                          4: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}}
    for i in range(0, 24):
        a = i // 5
        b = i % 5
        con.game_over(i)
        assert con.scores[a][b] == i
    con.game_over(24)
    assert con.scores == {0: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
                          1: {0: 5, 1: 6, 2: 7, 3: 8, 4: 9},
                          2: {0: 10, 1: 11, 2: 12, 3: 13, 4: 14},
                          3: {0: 15, 1: 16, 2: 17, 3: 18, 4: 19},
                          4: {0: 20, 1: 21, 2: 22, 3: 23, 4: 24}}
