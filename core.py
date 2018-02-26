import logging
import random
from uuid import uuid4

import tensorflow as tf


class Graph:
    def __init__(self, shape):
        """
        shape: dict{
            node_id: dict{
            'nodes': list[list[in_node_id, weight, bias]],
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

    def run(self, ins):
        logging.info('Compute Graph executed!')
        fetches = [i+':0' for i in self.exit_nodes]
        new_ins = {}
        for key in ins.keys():
            new_ins[key+':0'] = ins[key]

        if fetches:
            with tf.Session(graph=self.working) as sess:
                return (sess.run(fetches, new_ins), self.exit_nodes)

    def create_picture(self):
        with tf.Session(graph=self.working) as sess:
            writer = tf.summary.FileWriter('logs/', sess.graph)
            self.run({'a': 0.5, 'b': 0.4})
            writer.close()

    def create_working(self):
        logging.info('Creating workable compute graph!')
        work_nodes = {}
        exit_nodes = []
        for node in self.shape.keys():
            if self.shape[node]['type'] == 'exit':
                exit_nodes.append(node)

        def make_node(node):
            with self.working.as_default():
                enter_nodes = set(i[0] for i in self.shape[node]['nodes'])
                missing_nodes = enter_nodes - set(work_nodes.keys())
                for i_node in missing_nodes:
                    make_node(i_node)
                if self.shape[node]['type'] in ['hidden', 'exit']:
                    add_list = []
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
                            )
                        )
                    work_nodes[node] = tf.add_n(add_list, name=node)
                elif self.shape[node]['type'] == 'enter':
                    work_nodes[node] = tf.placeholder(tf.float32, name=node)
        with self.working.as_default():
            for exit_node in exit_nodes:
                make_node(exit_node)


class NeatGraph:
    def __init__(self, genes):
        """
        genotype: dict{
            'nodes': list[dict{'id': id, 'type': type}],
            'conns': dict{
                innov: dict{'in': in_node_id, 'out': out_node_id, 'weight': weight, 'enabled': enabled}
            }
        }
        phenotype: Graph
        """
        logging.info('NEAT Graph created!')
        self.genotype = genes
        self.phenotype = Graph(self._convert_genes_to_usable_format())

    def _convert_genes_to_usable_format(self):
        usable = {}
        for node in self.genotype['nodes']:
            logging.debug(f'node: {node}')
            ins = []
            for conn in self.genotype['conns'].values():
                logging.debug(f'\tconn: {conn}')
                if conn['out'] == node['id']:
                    ins.append([conn['in'], conn['weight'] if conn['enabled'] else 0, 0])
                    logging.debug(f'\t\tins: {ins}')
            usable[node['id']] = {'nodes': ins, 'type': node['type']}
            logging.debug(f'\tusable: {usable}')
        return usable

    def recalculate_phenotype(self):
        self.phenotype = Graph(self._convert_genes_to_usable_format())


class NeatController:
    def __init__(self, genera, species):
        self.genera_count = genera
        self.species_count = species
        self.graphs = {}
        self.scores = {}
        self.global_innov = 0
        self.current = (0, 0)
        for i in range(0, self.genera_count):
            self.graphs[i] = {}
            self.scores[i] = {}
            for j in range(0, self.species_count):
                self.graphs[i][j] = NeatGraph({})
                self.scores[i][j] = 0

    def run(self, inputs):
        return self.graphs[self.current[0]][self.current[1]].run(inputs)

    def game_over(self, score):
        self.scores[self.current[0]][self.current[1]] = score
        if self.current == (self.genera_count - 1, self.species_count - 1):
            self.current = (0, 0)
            self.breed()
        elif self.current[1] == self.species_count - 1:
            self.current = (self.current[0] + 1, 0)
        else:
            self.current = (self.current[0], self.current[1] + 1)

    def process_genera(self, new_graphs, genera):
        graph_copy = self.graphs.copy()
        top5 = {}
        new_graphs[genera] = {}
        child_counter = 0
        for i in range(0, 5):
            top = max(graph_copy[genera].keys, key=(lambda key: graph_copy[genera][key]))
            top5[self.scores[genera][top]] = self.graphs[genera][top]
            del graph_copy[genera][top]
        for graph_a in top5.keys():
            for graph_b in top5.keys():
                a_genotype = top5[graph_a].genotype
                b_genotype = top5[graph_b].genotype
                innov_counter = 1
                child_genotype = {'nodes': [], 'conns': {}}
                needed_nodes = set()
                while innov_counter <= self.global_innov:
                    a_gene = a_genotype['conns'].get(innov_counter)
                    b_gene = b_genotype['conns'].get(innov_counter)
                    if a_gene and b_gene:
                        choice = random.randint(1, 2)
                        if choice == 1:
                            needed_nodes += {a_gene['in'], a_gene['out']}
                            child_genotype['conns'][innov_counter] = a_gene
                        else:
                            needed_nodes += {b_gene['in'], b_gene['out']}
                            child_genotype['conns'][innov_counter] = b_gene
                    elif a_gene is None and b_gene:
                        if graph_b >= graph_a:
                            needed_nodes += {b_gene['in'], b_gene['out']}
                            child_genotype['conns'][innov_counter] = b_gene
                    elif a_gene and b_gene is None:
                        if graph_a >= graph_b:
                            needed_nodes += {a_gene['in'], a_gene['out']}
                            child_genotype['conns'][innov_counter] = a_gene
                    else:
                        break
                for node in a_genotype['nodes']:
                    if node['id'] in needed_nodes:
                        child_genotype['nodes'].append(node)
                        needed_nodes -= node['id']
                if needed_nodes:
                    for node in b_genotype['nodes']:
                        if node['id'] in needed_nodes:
                            child_genotype['nodes'].append(node)
                            needed_nodes -= node['id']
                if random.randint(0, 100) < 10:
                    """Mutate child"""
                    choice = random.randint(1, 2)
                    if choice == 1:
                        """Add node"""
                        conn = random.choice(child_genotype['conns'].keys())
                        child_genotype['conns'][conn]['enabled'] = False
                        node_id = uuid4()
                        conn1 = {'in': conn['in'], 'out': node_id, 'weight': conn['weight'], 'enabled': True}
                        conn2 = {'in': node_id, 'out': conn['out'], 'weight': conn['weight'], 'enabled': True}
                        child_genotype['conns'][self.global_innov + 1] = conn1
                        child_genotype['conns'][self.global_innov + 2] = conn2
                        child_genotype['nodes'].append({'id': node_id, 'type': 'hidden'})
                        self.global_innov += 2
                    elif choice == 2:
                        """Add connection"""
                        child_graph = NeatGraph(child_genotype)
                        child_shape = child_graph.phenotype.shape
                        out_node = random.choices(child_shape.keys())
                        while not out_node['nodes']:
                            out_node = random.choices(child_shape)

                        def find_before_nodes(node, possible_set=set()):
                            in_nodes = child_shape[node]
                            for inode in in_nodes:
                                possible_set += {inode[0]}
                                find_before_nodes(inode[0], possible_set)
                            return possible_set

                        possible_in_nodes = find_before_nodes(out_node)
                        in_node = random.choice(list(possible_in_nodes))
                        conn = {'in': in_node, 'out': out_node, 'weight': 0.5, 'enabled': True}
                        child_genotype['conns'][self.global_innov + 1] = conn
                        self.global_innov += 1
                new_graphs[genera][child_counter] = NeatGraph(child_genotype)
                child_counter += 1
                if child_counter == self.species_count - 1:
                    self.graphs = new_graphs
                    return new_graphs

    def breed(self):
        new_graphs = {}
        for genera in self.graphs.keys():
            new_graphs = self.process_genera(new_graphs, genera)
