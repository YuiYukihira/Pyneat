import tensorflow as tf
import logging

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