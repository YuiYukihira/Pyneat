import logging
import random
from uuid import uuid4
from typing import Dict, List, Union, Tuple
from dataclasses import dataclass
import multiprocessing
import dill

import tensorflow as tf
import os
import gc
## Define types

GRAPH_SHAPE = Dict[str, Dict[str, Union[str, List[Tuple[str, float, float]]]]]

## Define Constants

THREAD_COUNT = multiprocessing.cpu_count()-1

logging.basicConfig(filename='pyneat.log', level=logging.DEBUG)

class Graph:
    """A base graph, defines a shape and
    also creates and stores a tensorflow graph."""
    def __init__(self, shape: GRAPH_SHAPE):
        """
        shape: dict{
            node_id: dict{
            'nodes': list[tuple(in_node_id, weight, bias)],
            'type': node_type
            }
        }
        """
#        logging.info('Compute Graph Created!')
        self.shape = shape
        self.working = tf.Graph()
        self.exit_nodes = []
        self.create_working()
        for key in self.shape.keys():
            if self.shape[key]['type'] == 'exit':
                self.exit_nodes.append(key)

    def run(self, ins: Dict[str, Union[int, float]]) -> Tuple[List[float], List[str]]:
        """Takes a dictionary of input node name, and value.
        Returns a tuple of a list of output values,
        and a list of output nodes."""
#        logging.info('Compute Graph executed!')
        # Rename the node names so we get the tensorflow ID's
        # (this only works if the nodes have unique names.)
        fetches = [i+':0' for i in self.exit_nodes]
        new_ins = {}
        # Rename the node names so we get the tensorflow ID's
        # (this only works if the nodes have unique names.)
        for key in ins.keys():
            new_ins[key+':0'] = ins[key]

        if fetches:
            # Create a temporary tensorflow session called
            # sess which uses our graph.
            with tf.Session(graph=self.working) as sess:
                # Fetch and return the output.
                return (sess.run(fetches, new_ins), self.exit_nodes)

    def create_working(self):
        """Called internally,
        fills the Tensorflow Graph "working"
        with the data from "shape"."""
#        logging.info('Creating workable compute graph!')
        # A dictionary for storing our previously computed nodes.
        work_nodes = {}

        def make_node(node):
            """Recursive function for populating
            the graph with the information"""
            with self.working.as_default():
                # Find the nodes that enter the target node.
                enter_nodes = set(
                    i[0] for i in self.shape[node]['nodes'])
                # Find the nodes not yet in work_nodes.
                missing_nodes = enter_nodes - set(work_nodes.keys())
                for i_node in missing_nodes:
                    # Run this function on the missing nodes.
                    make_node(i_node)
                # If the node is a 'hidden' or 'exit' node.
                if self.shape[node]['type'] in ['hidden', 'exit']:
                    add_list = []
                    # Check that there nodes entering this node.
                    if enter_nodes != set():
                        # If there is, add a node into
                        # the graph with the previous node,
                        # the weight and bias.
                        for i_node in enter_nodes:
                            for n in self.shape[node]['nodes']:
                                if n[0] == i_node:
                                    weight = n[1]
                                    bias = n[2]
                                    break
                            add_list.append(
                                (
                                    work_nodes[i_node]*tf.constant(
                                        weight, dtype=tf.float32
                                    )
                                )+tf.constant(
                                    bias, dtype=tf.float32
                                ))
                    else:
                        # If there isn't an input node.
                        # create a fake input that outputs zero
                        # as tensorflow requires a tensor
                        # to have an input.
                        add_list.append(
                            tf.constant(0.0, dtype=tf.float32))
                    # Create a new tensor that sums all
                    # the nodes together and add it to
                    # work_nodes with the id as the key.
                    work_nodes[node] = tf.add_n(add_list, name=node)
                # If the node if type 'enter'
                elif self.shape[node]['type'] == 'enter':
                    # Create a new placeholder
                    # tensor and add it to work_nodes
                    # with the id as the key.
                    work_nodes[node] = tf.sigmoid(
                        tf.placeholder(tf.float32, name=node))
        with self.working.as_default():
            for node in self.shape:
                make_node(node) # Make every node in the shape.

@dataclass
class NodeGene:
    """
    Holds information about a node gene.
    id: node id. (a string)
    type: node type. (one of: "enter", "hidden", "exit")
    """

    id: str
    type: str

    __slots__ = ["id", "type"]

    def __post_init__(self):
        if not isinstance(self.id, str):
            raise TypeError
        if not isinstance(self.type, str):
            raise TypeError
        if not self.type in ['enter', 'hidden', 'exit']:
            raise ValueError

@dataclass
class ConnGene:
    """
    Holds information about a connection gene.
    in_node: the input node. (a string)
    out_node: the output node. (a string)
    weight: the connection weight. (a float)
    enabled: whether the connection is enabled. (a bool)
    """

    in_node: str
    out_node: str
    weight: float
    enabled: bool

    __slots__ = ["in_node", "out_node", "weight", "enabled"]

    def __post_init__(self):
        if not isinstance(self.in_node, str):
            raise TypeError
        if not isinstance(self.out_node, str):
            raise TypeError
        if not isinstance(self.weight, float):
            raise TypeError
        if not isinstance(self.enabled, bool):
            raise TypeError

@dataclass
class Genotype:
    """
    Holds information on a genotype.
    nodes: the node genes. (a list of NodeGene instances)
    conns: the connection genes.
    (a dictionary of key innovation score (int)
    and value ConnGene instances)
    mutation_rate: determines how likely this graph is to mutate.
    """

    nodes: List[NodeGene]
    conns: Dict[int, ConnGene]
    mutation_rate: int

    __slots__ = ['nodes', 'conns', 'mutation_rate']

    def __post_init__(self):
        if not isinstance(self.nodes, list):
            raise TypeError
        else:
            for i in self.nodes:
                if not isinstance(i, NodeGene):
                    raise TypeError
        if not isinstance(self.conns, Dict):
            raise TypeError
        else:
            for i, j in self.conns.items():
                if not isinstance(i, int):
                    raise TypeError
                if not isinstance(j, ConnGene):
                    raise TypeError
        if not isinstance(self.mutation_rate, int):
            raise TypeError

class NeatGraph:
    """
    A graph that has a genotype that follows the NEAT style
    and a phenotype that uses tensorflow for computation.
    """
    def __init__(self, genes: Genotype):
#        logging.info('NEAT Graph created!')
        if not isinstance(genes, Genotype):
            raise TypeError
        self.genotype = genes
        # create our phenotype from our genes.
        self.phenotype = Graph(
            self.convert_genes_to_usable_format(self.genotype))

    @staticmethod
    def convert_genes_to_usable_format(genotype: Genotype) -> GRAPH_SHAPE:
        """The information in the Genotype class is not in the correct format
        so we have to convert it for the graph class."""
        if not isinstance(genotype, Genotype):
            raise TypeError
        usable = {} # Dictionary of previously computed nodes.
        for node in genotype.nodes: # For every node in the genotype.
            #logging.debug(f'node: {node}')
            ins = [] # List of nodes that input into this one.
            # For all the values in the genotype connections.
            for conn in genotype.conns.values():
                #logging.debug(f'\tconn: {conn}')
                # Does this connection have the node as it's output node?
                try:
                    if conn.out_node == node.id:
                        # Add the input node into in's with the
                        # id and weight. And a bias of 0 as we don't use it.
                        ins.append((conn.in_node, conn.weight if conn.enabled else 0, 0))
                        #logging.debug(f'\t\tins: {ins}')
                except AttributeError as e:
                    print(e)
                    print(f"conn: {conn}\nnode: {node}")
            # Add the input nodes to the useable dict with the node id as the key.
            usable[node.id] = {'nodes': ins, 'type': node.type}
            #logging.debug(f'\tusable: {usable}')
        return usable # Return the usable genes.

    def recalculate_phenotype(self):
        """Recalculate the phenotype if needed (it shouldn't be)."""
        self.phenotype = Graph(self.convert_genes_to_usable_format(self.genotype))

    def run(self, ins):
        """Return the output from phenotype.run"""
        return self.phenotype.run(ins)


class BreedController:
    __slots__ = ['required_nodes', 'scores',
                 'graphs', 'species_count', 'genera_count']
    def __init__(self, required_nodes, scores,
                 graphs, species_count, genera_count):
        self.required_nodes = required_nodes
        self.scores = scores
        self.graphs = graphs
        self.species_count = species_count
        self.genera_count = genera_count

    def run(self, args):
        genera = args[0]
        innovation_dict = args[1]
        innovation_lock = args[2]
        logging.debug(
            f'##Starting process {os.getpid()} for genera: {genera}')
        top5 = {}
        new_graphs = {}
        child_counter = 0
        graphs_copy = self.scores.copy()

        # Find the top 5 scoring graphs in the genera.
        for i in range(0, 5):
            top = max(graphs_copy[genera].keys(),
                      key=(lambda key: graphs_copy[genera][key]))
            top5[top] = (
                self.graphs[genera][top],
                self.scores[genera][top])
            del graphs_copy[genera][top]
        # Repeat for each graph in the top 5.
        for a_tuple in top5.values():
            # Make Graph A.
            a_graph = a_tuple[0]
            a_score = a_tuple[1]
            # Repeat for each graph in the top 5.
            for b_tuple in top5.values():
                # Make the Graph B.
                b_graph = b_tuple[0]
                b_score = b_tuple[1]
                # Only breed if the two graphs are different.
                if a_graph is not b_graph:
                    a_genotype = a_graph
                    b_genotype = b_graph
                    # Get the child's mutation rate
                    # from the most fit parent.
                    # If they are the same fitness,
                    # select randomly.
                    if a_score > b_score:
                        mutation_rate = a_genotype.mutation_rate
                    elif b_score > a_score:
                        mutation_rate = b_genotype.mutation_rate
                    else:
                        mutation_rate = random.choice(
                            [b_genotype.mutation_rate,
                             a_genotype.mutation_rate])
                    # Create and empty genotype C for the child.
                    c_genotype = Genotype([], {}, mutation_rate)
                    # Find the nodes that are required as a bare minimum
                    # and add them to the set of needed nodes.
                    needed_nodes = set(self.required_nodes.keys())
                    # Repeat until we reach the
                    # global innovation score counter
                    try:
                        max_innov = max(innovation_dict.keys())
                    except ValueError:
                        max_innov = 0
                    for innov_counter in range(0, max_innov):
                        a_gene = a_genotype.conns.get(innov_counter)
                        b_gene = b_genotype.conns.get(innov_counter)
                        if a_gene is not None and b_gene is not None:
                            # Gene present in both parents.
                            if a_score > b_score:
                                # A is fitter, so inherit gene from a.
                                needed_nodes |= {a_gene.in_node,
                                                 a_gene.out_node}
                                c_genotype.conns[innov_counter] = a_gene
                            elif b_score > a_score:
                                # B is fitter, so inherit gene from b.
                                needed_nodes |= {b_gene.in_node,
                                                 b_gene.out_node}
                                c_genotype.conns[innov_counter] = b_gene
                            else:
                                # A is as fit as B so select randomly.
                                gene = random.choice([a_gene, b_gene])
                                needed_nodes |= {gene.in_node,
                                                 gene.out_node}
                                c_genotype.conns[innov_counter] = gene
                        elif a_gene and b_gene is None:
                            # gene not present in b, so inherit from a.
                            needed_nodes |= {a_gene.in_node,
                                             a_gene.out_node}
                            c_genotype.conns[innov_counter] = a_gene
                        elif b_gene and a_gene is None:
                            # gene not present in a, so inherit from b.
                            needed_nodes |= {b_gene.in_node,
                                             b_gene.out_node}
                            c_genotype.conns[innov_counter] = b_gene
                    # Add the node genes with information
                    # if they are from the required nodes.
                    for node_id, node_type in self.required_nodes.items():
                        if node_id in needed_nodes:
                            c_genotype.nodes.append(
                                NodeGene(node_id, node_type))
                            needed_nodes -= {node_id}
                    if needed_nodes: # If there are still nodes left.
                        # Add the node genes with
                        # information from genotype A.
                        for node in a_genotype.nodes: #
                            if node.id in needed_nodes:
                                c_genotype.nodes.append(node)
                                needed_nodes -= {node.id}
                    if needed_nodes: # If there are still nodes left.
                        # Add the node genes with
                        # information from genotype B.
                        for node in b_genotype.nodes:
                            if node.id in needed_nodes:
                                c_genotype.nodes.append(node)
                                needed_nodes -= {node.id}
                    # Mutate the C genotype if it passes the check.
                    if random.randint(0, 100) < c_genotype.mutation_rate:
                        choice = random.randint(0,4)
                        if choice == 0:
                            # Change mutation rate
                            c_genotype.mutation_rate = random.randint(
                                0,100)
                        elif choice == 1:
                            # Add a new node.
                            if c_genotype.conns != {}:
                                conn = random.choice(
                                    list(c_genotype.conns.keys()))
                                c_genotype.conns[conn].enabled = False
                                node_id = str(uuid4())
                                conn1 = ConnGene(
                                    c_genotype.conns[conn].in_node,
                                    node_id, 1.0, True)
                                conn2 = ConnGene(
                                    node_id,
                                    c_genotype.conns[conn].out_node,
                                    c_genotype.conns[conn].weight,
                                    c_genotype.conns[conn].enabled)
                                innovation_lock.acquire()
                                try:
                                    innovation = max(innovation_dict.keys())
                                except ValueError:
                                    innovation = 0
                                c_genotype.conns[innovation + 1] = conn1
                                innovation_dict[innovation + 1] = {'in': conn1.in_node, 'out': conn1.out_node}
                                c_genotype.conns[innovation + 2] = conn2
                                innovation_dict[innovation + 2] = {'in': conn2.in_node, 'out': conn2.out_node}
                                innovation_lock.release()
                                c_genotype.nodes.append(
                                    NodeGene(node_id, "hidden"))
                        elif choice == 2:
                            # Add a new connection
                            c_shape = NeatGraph.convert_genes_to_usable_format(
                                c_genotype)
                            possible_keys = set(c_shape.keys())
                            out_node = list(possible_keys)[
                                random.randint(0,len(possible_keys)-1)]
                            while c_shape[out_node]['type'] == 'enter':
                                possible_keys -= {out_node}
                                out_node = list(possible_keys)[
                                    random.randint(
                                        0,len(possible_keys)-1)]
                            def find_possible_nodes(shape, node):
                                def find_after_nodes(
                                        shape, node,
                                        possible_set=None):
                                    if possible_set is None:
                                        possible_set = set()
                                    possible_set |= {node}
                                    for inode_name, inode_val in shape.items():
                                        in_nodes = {
                                            i[0] for i in inode_val['nodes']}
                                        if node in in_nodes:
                                            possible_set |= find_after_nodes(
                                                shape, inode_name, possible_set)
                                    return possible_set
                                impossible_set = find_after_nodes(shape, node)
                                nodes_set = {gene.id for gene in c_genotype.nodes}
                                possible_set = nodes_set - impossible_set
                                return possible_set

                            possible_in_nodes = find_possible_nodes(c_shape, out_node)
                            in_node = random.choice(list(possible_in_nodes))
                            w_sign = random.randint(1,2)
                            w_amount = random.random()*2
                            innovation_lock.acquire()
                            innovation = None
                            for innov, gene in innovation_dict.items():
                                if gene['in'] == in_node and gene['out'] == out_node:
                                    innovation = innov
                                    break
                            if not innovation:
                                try:
                                    innovation = max(innovation_dict.keys()) + 1
                                except ValueError:
                                    innovation = 0
                                innovation_dict[innovation] = {'in': in_node, 'out': out_node}
                            innovation_lock.release()
                            conn = ConnGene(
                                in_node,
                                out_node,
                                w_amount if w_sign == 1 else -w_amount, True)

                            c_genotype.conns[innovation] = conn
                        elif choice == 3:
                            # Toggle connection enabled
                            if c_genotype.conns != {}:
                                conn = random.choice(
                                    list(c_genotype.conns.keys()))
                                c_genotype.conns[conn].enabled = not c_genotype.conns[
                                    conn].enabled
                        elif choice == 4:
                            # Change connection weight
                            if c_genotype.conns != {}:
                                conn = random.choice(
                                    list(c_genotype.conns.keys()))
                                sign = random.randint(1,2)
                                amount = random.random()*2
                                c_genotype.conns[conn].weight = amount if sign == 1 else -amount
                    new_graphs[child_counter] = c_genotype
                    child_counter += 1
                    # If we've hit the amount of children needed.
                    if child_counter == self.species_count:
                        logging.debug(f'##Ending proccess {os.getpid()}\
 for genera: {genera}')
                        # return the new graphs for this genera.
                        return [genera, new_graphs]

@dataclass
class NeatSave(object):
    """A save class for a NEAT controller"""
    genera_count: int
    species_count: int
    graphs: Dict[int, Dict[int, Genotype]]
    scores: Dict[int, Dict[int, int]]
    current: Tuple[int, int]
    required_nodes: Dict[str, str]
    innovation_dict: dict
    __slots__ = [
        'genera_count',
        'species_count',
        'graphs',
        'scores',
        'current',
        'required_nodes',
        'innovation_dict'
    ]

    def __post_init__(self):
        if not isinstance(self.genera_count, int):
            raise TypeError
        if not isinstance(self.species_count, int):
            raise TypeError
        if not isinstance(self.graphs, dict):
            raise TypeError
        else:
            for i, j in self.graphs.items():
                if not isinstance(i, int):
                    raise TypeError
                if not isinstance(j, dict):
                    raise TypeError
                else:
                    for k, l in j.items():
                        if not isinstance(k, int):
                            raise TypeError
                        if not isinstance(l, Genotype):
                            raise TypeError

class NeatController(object):
    """Controlls the NEAT process."""
    __slots__ = [
        'genera_count',
        'species_count',
        'graphs',
        'scores',
        'current',
        'required_nodes',
        'bstart_callbacks',
        'bend_callbacks',
        'bstatus_callbacks',
        'innovation_dict',
        'innovation_lock']
    def __init__(
            self,
            genera: int,
            species: int,
            required_nodes: Dict[str, str]):
        if not isinstance(genera, int):
            raise TypeError
        if not isinstance(species, int):
            raise TypeError
        if not isinstance(required_nodes, dict):
            raise TypeError
        else:
            for i, j in required_nodes.items():
                if not isinstance(i, str):
                    raise TypeError
                if not isinstance(j, str):
                    raise TypeError
                elif j not in ["enter", "hidden", "exit"]:
                    raise ValueError
        self.genera_count = genera
        self.species_count = species
        self.graphs = {}
        self.scores = {}
        self.current = (0, 0)
        self.required_nodes = required_nodes
        self.bstart_callbacks = []
        self.bend_callbacks = []

        manager = multiprocessing.Manager()
        self.innovation_dict = manager.dict()
        self.innovation_lock = manager.RLock()
        for i in range(0, self.genera_count):
            self.graphs[i] = {}
            self.scores[i] = {}
            for j in range(0, self.species_count):
                self.graphs[i][j] = NeatGraph(
                    Genotype(
                        [NodeGene(i, j) for i,
                         j in self.required_nodes.items()],
                    {},
                    random.randint(0,100)))
                self.scores[i][j] = 0

    def save_state(self, filename: str ="NEATsave.pkl") -> None:
        ns = NeatSave(
            self.genera_count,
            self.species_count,
            {i: {ji: jk.genotype for ji, jk in j.items()} for i, j in self.graphs.items()},
            self.scores,
            self.current,
            self.required_nodes,
            self.innovation_dict
        )
        with open(filename, 'wb') as output:
            dill.dump(ns, output)


    @classmethod
    def load_state(cls, filename: str ="NEATsave.pkl"):
        with open(filename, 'rb') as input_file:
            obj = dill.load(input_file)
        inst = cls(0, 0, obj.required_nodes)
        inst.genera_count = obj.genera_count
        inst.species_count = obj.species_count
        inst.graphs = {i: {ji: NeatGraph(jk) for ji, jk in j.items()} for i, j in obj.graphs.items()}
        inst.scores = obj.scores
        inst.current = obj.current
        for i, j in obj.innovation_dict.items():
            inst.innovation_dict[i] = j
        return inst

    @property
    def current_graph(self):
        return self.graphs[self.current[0]][self.current[1]]

    def run(self, inputs):
        """Return the output of the current graph's run method."""
        return self.graphs[self.current[0]][self.current[1]].run(inputs)

    def game_over(self, score):
        """assign score to the current graph and move onto the next one.
        If no graphs left, call the breed method."""
        self.scores[self.current[0]][self.current[1]] = score
        if self.current == (
                self.genera_count - 1,
                self.species_count - 1):
            self.current = (0, 0)
            self.breed()
        elif self.current[1] == self.species_count - 1:
            self.current = (self.current[0] + 1, 0)
        else:
            self.current = (self.current[0], self.current[1] + 1)

    def add_breed_start_callback(self, cb):
        """Add a function to be called then the breeding is started"""
        self.bstart_callbacks.append(cb)

    def add_breed_end_callback(self, cb):
        """Add a funciton to be called when the breeding is finished."""
        self.bend_callbacks.append(cb)

    def breed(self):
        for cb in self.bstart_callbacks: # Run all our start callbacks,
            cb(self)
        new_graphs = {} # Create a new dict for our new graphs.
        # Create a new pool for our processes.
        pool = multiprocessing.Pool(THREAD_COUNT)
        ins = [[i, self.innovation_dict, self.innovation_lock] for i in range(0, self.genera_count)]
        bController = BreedController(
            self.required_nodes,
            self.scores,
            {i:
             {ji: jk.genotype for ji, jk in j.items()
             } for i, j in self.graphs.items()},
            self.species_count,
            self.genera_count
        )
        # Get our results from our pool.
        results = pool.map(bController.run, ins)
        pool.close()
        pool.join()
        for result in results:
            # Add our results to new_graphs
            # converting the genotypes into NeatGraphs.
            new_graphs[result[0]] = {
                i: NeatGraph(j) for i, j in result[1].items()}
        self.graphs = new_graphs # replace the graphs with the new ones.
        for cb in self.bend_callbacks: # Run our end callbacks.
            cb(self)
        gc.collect()
