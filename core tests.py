import logging

import core

logging.basicConfig(filename='CoreTests.log', level=logging.INFO)


def test1():
    graph = core.Graph({
        'e': {'nodes': [['d', 1, 0]], 'type': 'exit'},
        'd': {'nodes': [], 'type': 'enter'},
        'c': {'nodes': [['a', 1, 0], ['b', 1, 0]], 'type': 'exit'},
        'b': {'nodes': [], 'type': 'enter'},
        'a': {'nodes': [], 'type': 'enter'}
    })
    logging.debug(graph.run({'a': 1, 'b': 1, 'd': 1}))
    if graph.run({'a': 1, 'b': 1, 'd': 1}) == ([1.0, 2.0], ['e', 'c']):
        logging.info('Test 1: Passed.')
    else:
        logging.info('Test 1: Failed!')


def test2():
    graph = core.NeatGraph({
        'nodes': [
            {'id': 'e', 'type': 'exit'},
            {'id': 'd', 'type': 'enter'},
            {'id': 'c', 'type': 'exit'},
            {'id': 'b', 'type': 'enter'},
            {'id': 'a', 'type': 'enter'},
            ],
        'conns': {
            1: {'in': 'a', 'out': 'c', 'weight': 1, 'enabled': True},
            2: {'in': 'b', 'out': 'c', 'weight': 1, 'enabled': True},
            3: {'in': 'd', 'out': 'e', 'weight': 1, 'enabled': True}
        }
    })
    logging.debug(graph.phenotype.run({'a': 1, 'b': 1, 'd': 1}))
    if graph.phenotype.run({'a': 1, 'b': 1, 'd': 1}) == ([1.0, 2.0], ['e', 'c']):
        logging.info('Test 2: Passed.')
    else:
        logging.info('Test 2: Failed!')


def test3():
    graph = core.Graph({})
    logging.debug(graph.run({}))
    logging.debug(graph.run({'a': 1}))
    if graph.run({}) is None and graph.run({'a': 1}) is None:
        logging.info('Test 3: Passed.')
    else:
        logging.info('Test 3: Failed!')


if __name__ == '__main__':
    test1()
    test2()
    test3()
