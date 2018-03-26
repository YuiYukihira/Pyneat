import logging
import random
from uuid import uuid4
from typing import Dict, List, Union, Tuple

import tensorflow as tf

## Define types

GRAPH_SHAPE = Dict[str, Dict[str, Union[str, List[Tuple[str, float, float]]]]]

class Graph:
    """A base graph, defines a shape and also creates and stores a tensorflow graph."""
    def __init__(self, shape: GRAPH_SHAPE):
        """
        shape: dict{
            node_id: dict{
            'nodes': list[tuple(in_node_id, weight, bias)],
            'type': node_type
            }
        }
        """
        logging.info('Compute Graph Created!')
        self.shape = shape
        self.working = tf.Graph()
        self.exit_nodes = []
        self.create_working()
        for key in self.shape.keys():
            if self.shape[key]['type'] == 'exit':
                self.exit_nodes.append(key)

    def run(self, ins: Dict[str, Union[int, float]]) -> Tuple[List[float], List[str]]:
        """Takes a dictionary of input node name, and value. Returns a tuple of a list of output values, and a list of output nodes."""
        logging.info('Compute Graph executed!')
        fetches = [i+':0' for i in self.exit_nodes] # Rename the node names so we get the tensorflow ID's (this only works if the nodes have unique names.)
        new_ins = {}
        for key in ins.keys():
            new_ins[key+':0'] = ins[key] # Rename the node names so we get the tensorflow ID's (this only works if the nodes have unique names.)

        if fetches:
            with tf.Session(graph=self.working) as sess: # Create a temporary tensorflow session called sess which uses our graph.
                return (sess.run(fetches, new_ins), self.exit_nodes) # Fetch and return the output.

    def create_picture(self):
        """Writes a picute of the graph for TensorBoard to use."""
        with tf.Session(graph=self.working) as sess:
            writer = tf.summary.FileWriter('logs/', sess.graph)
            self.run({'a': 0.5, 'b': 0.4})
            writer.close()

    def create_working(self):
        """Called internally, fills the Tensorflow Graph "working" with the data from "shape"."""
        logging.info('Creating workable compute graph!')
        work_nodes = {} # A dictionary for storing our previously computed nodes.

        def make_node(node):
            """Recursive function for populating the graph with the information"""
            with self.working.as_default():
                enter_nodes = set(i[0] for i in self.shape[node]['nodes']) # Find the nodes that enter the target node.
                missing_nodes = enter_nodes - set(work_nodes.keys()) # Find the nodes not yet in work_nodes.
                for i_node in missing_nodes:
                    make_node(i_node) # Run this function on the missing nodes.
                if self.shape[node]['type'] in ['hidden', 'exit']: # If the node is a 'hidden' or 'exit' node.
                    add_list = []
                    if enter_nodes != set(): # Check that there nodes entering this node.
                        for i_node in enter_nodes: # If there is, add a node into the graph with the previous node, the weight and bias.
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
                        add_list.append(tf.constant(0.0, dtype=tf.float32)) # If there isn't an input node. create a fake input that outputs zero as tensorflow requires a tensor to have an input.
                    work_nodes[node] = tf.add_n(add_list, name=node) # Create a new tensor that sums all the nodes together and add it to work_nodes with the id as the key.
                elif self.shape[node]['type'] == 'enter': # If the node if type 'enter'
                    work_nodes[node] = tf.sigmoid(tf.placeholder(tf.float32, name=node)) # Create a new placeholder tensor and add it to work_nodes ith the id as the key.
        with self.working.as_default():
            for node in self.shape:
                make_node(node) # Make every node in the shape.

class NodeGene:
    """
    Holds information about a node gene.
    id: node id. (a string)
    type: node type. (one of: "enter", "hidden", "exit")
    """
    __slots__ = ["id", "type"]
    def __init__(self, node_id: str, node_type: str):
        self.id = node_id
        self.type = node_type

    def __str__(self):
        return f'id: {self.id}, type: {self.type}'

class ConnGene:
    """
    Holds information about a connection gene.
    in_node: the input node. (a string)
    out_node: the output node. (a string)
    weight: the connection weight. (a float)
    enabled: whether the connection is enabled. (a bool)
    """
    __slots__ = ["in_node", "out_node", "weight", "enabled"]
    def __init__(self, in_node: str, out_node: str, weight: float, enabled: bool):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled

    def __str__(self):
        return f'in: {self.in_node}, out: {self.out_node}, weight: {self.weight}, enabled: {self.enabled}'



class Genotype:
    """
    Holds information on a genotype.
    nodes: the node genes. (a list of NodeGene instances)
    conns: the connection genes. (a dictionary of key innovation score (int) and value ConnGene instances)
    mutation_rate: determines how likely this graph is to mutate.
    """
    __slots__ = ['nodes', 'conns', 'mutation_rate']
    def __init__(self, nodes: List[NodeGene], conns: Dict[int, ConnGene], mutation_rate: int):
        self.nodes = nodes
        self.conns = conns
        self.mutation_rate = mutation_rate

    def __str__(self):
        nodes = '\n'.join('\t'+str(i) for i in self.nodes)
        conns = '\n'.join('\t'+str(i) for i in self.conns.values())
        string = f'nodes: {nodes}\nconns: {conns}\nmutation rate: {self.mutation_rate}'
        return string

class NeatGraph:
    """
    A graph that has a genotype that follows the NEAT style and a phenotype that uses tensorflow for computation.
    """
    def __init__(self, genes: Genotype):
        logging.info('NEAT Graph created!')
        self.genotype = genes
        self.phenotype = Graph(self.convert_genes_to_usable_format(self.genotype)) # create our phenotype from our genes.

    @staticmethod
    def convert_genes_to_usable_format(genotype: Genotype) -> GRAPH_SHAPE:
        """The information in the Genotype class is not in the correct format so we have to convert it for the graph class."""
        usable = {} # Dictionary of previously computed nodes.
        for node in genotype.nodes: # For every node in the genotype.
            logging.debug(f'node: {node}')
            ins = [] # List of nodes that input into this one.
            for conn in genotype.conns.values(): # For all the values in the genotype connections.
                logging.debug(f'\tconn: {conn}')
                if conn.out_node == node.id: # Does this connection have the node as it's output node?
                    ins.append((conn.in_node, conn.weight if conn.enabled else 0, 0)) # Add the input node into in's with the id and weight and a bias of 0 as we don't use them.
                    logging.debug(f'\t\tins: {ins}')
            usable[node.id] = {'nodes': ins, 'type': node.type} # Add the input nodes to the useable dict with the node id as the key.
            logging.debug(f'\tusable: {usable}')
        return usable # Return the usable genes.

    def recalculate_phenotype(self):
        """Recalculate the phenotype if needed (it shouldn't be)."""
        self.phenotype = Graph(self.convert_genes_to_usable_format(self.genotype))

    def run(self, ins):
        """Return the output from phenotype.run"""
        return self.phenotype.run(ins)



class NeatController:
    """Controlls the NEAT process."""
    def __init__(self, genera: int, species: int, required_nodes: Dict[str, str]):
        self.genera_count = genera
        self.species_count = species
        self.graphs = {}
        self.scores = {}
        self.global_innov = 0
        self.current = (0, 0)
        self.required_nodes = required_nodes
        for i in range(0, self.genera_count):
            self.graphs[i] = {}
            self.scores[i] = {}
            for j in range(0, self.species_count):
                self.graphs[i][j] = NeatGraph(Genotype([
                    NodeGene(i, j) for i, j in self.required_nodes.items()],{},random.randint(0,100)))
                self.scores[i][j] = 0

    def run(self, inputs):
        """Return the output of the current graph's run method."""
        return self.graphs[self.current[0]][self.current[1]].run(inputs)

    def game_over(self, score):
        """assign score to the current graph and move onto the next one. If no graphs left, call the breed method."""
        self.scores[self.current[0]][self.current[1]] = score
        if self.current == (self.genera_count - 1, self.species_count - 1):
            self.current = (0, 0)
            self.breed()
        elif self.current[1] == self.species_count - 1:
            self.current = (self.current[0] + 1, 0)
        else:
            self.current = (self.current[0], self.current[1] + 1)

    def _process_genera(self, genera):
        graphs_copy = self.scores.copy()
        top5 = {}
        new_graphs = {}
        child_counter = 0

        for i in range(0, 5): # Find the top 5 scoring graphs in the genera.
            top = max(graphs_copy[genera].keys(), key=(lambda key: graphs_copy[genera][key]))
            top5[top] = (self.graphs[genera][top], self.scores[genera][top])
            del graphs_copy[genera][top]
        for a_tuple in top5.values(): # Repeat for each graph in the top 5.
            # Make Graph A.
            a_graph = a_tuple[0]
            a_score = a_tuple[1]
            for b_tuple in top5.values(): # Repeat for each graph in the top 5.
                # Make the Graph B.
                b_graph = b_tuple[0]
                b_score = b_tuple[1]
                if a_graph != b_graph: # Only breed if the two graphs are different.
                    a_genotype = b_graph.genotype
                    b_genotype = b_graph.genotype
                    if a_score > b_score: # Get the child's mutation rate from the most fit parent. If they are the same fitness, select randomly.
                        mutation_rate = a_genotype.mutation_rate
                    elif b_score > a_score:
                        mutation_rate = b_genotype.mutation_rate
                    else:
                        mutation_rate = random.choice([b_genotype.mutation_rate, a_genotype.mutation_rate])
                    c_genotype = Genotype([], {}, mutation_rate) # Create and empty genotype C for the child.
                    needed_nodes = set(self.required_nodes.keys()) # Find the nodes that are required as a bare minimum and add them to the set of needed nodes.
                    for innov_counter in range(0, self.global_innov): # Repeat until we reach the global innovation score counter
                        a_gene = a_genotype.conns.get(innov_counter)
                        b_gene = b_genotype.conns.get(innov_counter)
                        if a_gene and b_gene:
                            # Gene present in both parents.
                            if a_score > b_score:
                                # A is fitter, so inherit gene from a.
                                needed_nodes |= {a_gene.in_node, a_gene.out_node}
                                c_genotype.conns[innov_counter] = a_gene
                            elif b_score > a_score:
                                # B is fitter, so inherit gene from b.
                                needed_nodes |= {b_gene.in_node, b_gene.out_node}
                                c_genotype.conns[innov_counter] = b_gene
                            else:
                                # A is as fit as B so select randomly.
                                gene = random.choice([a_gene, b_gene])
                                needed_nodes |= {gene.in_node, gene.out_node}
                                c_genotype.conns[innov_counter] = gene
                        elif a_gene and b_gene is None:
                            # gene not present in b, so inherit from a.
                            needed_nodes |= {a_gene.in_node, a_gene.out_node}
                            c_genotype.conns[innov_counter] = gene
                        elif b_gene and a_gene is None:
                            # gene not present in a, so inherit from b.
                            needed_nodes |= {b_gene.in_node, b_gene.out_node}
                            c_genotype.conns[innov_counter] = genera
                    for node_id, node_type in self.required_nodes.items(): # Add the node genes with information if they are from the required nodes.
                        if node_id in needed_nodes:
                            c_genotype.nodes.append(NodeGene(node_id, node_type))
                            needed_nodes -= {node_id}
                    if needed_nodes: # If there are still nodes left.
                        for node in a_genotype.nodes: # Add the node genes with information from genotype A.
                            if node.id in needed_nodes:
                                c_genotype.nodes.append(node)
                                needed_nodes -= {node.id}
                    if needed_nodes: # If there are still nodes left.
                        for node in b_genotype.nodes: # Add the node genes with information from genotype B.
                            if node.id in needed_nodes:
                                c_genotype.nodes.append(node)
                                needed_nodes -= {node.id}
                    if random.randint(0, 100) < c_genotype.mutation_rate: # Mutate the C genotype if it passes the check.
                        choice = random.randint(0,4)
                        if choice == 0:
                            # Change mutation rate
                            c_genotype.mutation_rate = random.randint(0,100)
                        elif choice == 1:
                            # Add a new node.
                            if c_genotype.conns != {}:
                                conn = random.choice(list(c_genotype.conns.keys()))
                                c_genotype.conns[conn].enabled = False
                                node_id = str(uuid4())
                                conn1 = ConnGene(c_genotype.conns[conn].in_node, node_id, 1, True)
                                conn2 = ConnGene(node_id, c_genotype.conns[conn].out_node, c_genotype.conns[conn].weight, c_genotype.conns[conn].enabled)
                                c_genotype.conns[self.global_innov + 1] = conn1
                                c_genotype.conns[self.global_innov + 2] = conn2
                                c_genotype.nodes.append(NodeGene(node_id, "hidden"))
                                self.global_innov += 2
                        elif choice == 2:
                            # Add a new connection
                            c_shape = NeatGraph.convert_genes_to_usable_format(c_genotype)
                            #out_node = random.choice(list(c_shape.keys()))
                            possible_keys = set(c_shape.keys())
                            out_node = list(possible_keys)[random.randint(0,len(possible_keys)-1)]
                            while c_shape[out_node]['type'] == 'enter':
                                #out_node = random.choice(list(c_shape.keys()))
                                possible_keys -= {out_node}
                                out_node = list(possible_keys)[random.randint(0,len(possible_keys)-1)]
                            def find_possible_nodes(shape, node):
                                def find_after_nodes(shape, node, possible_set=None):
                                    if possible_set is None:
                                        possible_set = set()
                                    possible_set |= {node}
                                    for inode_name, inode_val in shape.items():
                                        in_nodes = {i[0] for i in inode_val['nodes']}
                                        if node in in_nodes:
                                            possible_set |= find_after_nodes(shape, inode_name, possible_set)
                                    return possible_set
                                impossible_set = find_after_nodes(shape, node)
                                nodes_set = {gene.id for gene in c_genotype.nodes}
                                possible_set = nodes_set - impossible_set
                                return possible_set

                            possible_in_nodes = find_possible_nodes(c_shape, out_node)
                            in_node = random.choice(list(possible_in_nodes))
                            w_sign = random.randint(1,2)
                            w_amount = random.random()*2
                            conn = ConnGene(in_node, out_node, w_amount if w_sign == 1 else -w_amount, True)
                            c_genotype.conns[self.global_innov + 1] = conn
                            self.global_innov += 1
                        elif choice == 3:
                            # Toggle connection enabled
                            if c_genotype.conns != {}:
                                conn = random.choice(list(c_genotype.conns.keys()))
                                c_genotype.conns[conn].enabled = not c_genotype.conns[conn].enabled
                        elif choice == 4:
                            # Change connection weight
                            if c_genotype.conns != {}:
                                conn = random.choice(list(c_genotype.conns.keys()))
                                sign = random.randint(1,2)
                                amount = random.random()*2
                                c_genotype.conns[conn].weight = amount if sign == 1 else -amount
                    new_graphs[child_counter] = NeatGraph(c_genotype) # Make a NeatGraph from the genotype and add it test the list.
                    child_counter += 1
                    if child_counter == self.species_count: # If we've hit the amount of children needed.
                        return new_graphs # return the new graphs for this genera.
    def breed(self):
        new_graphs = {}
        for genera in self.graphs: # For every genera.
            n = self._process_genera(genera) # Get the new graphs for this genera.
            new_graphs[genera] = n # Set the genera to the new values
        self.graphs = new_graphs # replace the graphs with the new ones.
