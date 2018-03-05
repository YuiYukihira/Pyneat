import core
import random

training_list = [
    [{'a': 0, 'b': 0}, 0.0],
    [{'a': 0, 'b': 1}, 1.0],
    [{'a': 1, 'b': 0}, 1.0],
    [{'a': 1, 'c': 1}, 0.0]
]


def score_graphs(controller):
    for genera in range(0, controller.genera_count):
        for species in range(0, controller.species_count):
            combined_score = 0
            for train in training_list:
                result = controller.run(train[0])
                score = int((1 - (train[1] - result[0][0])) * 100)
                combined_score += score
            controller.game_over(combined_score)
        print('finished breeding')


def train(cycles, controller):
    for i in range(0, cycles):
        print('=================================================================================================')
        score_graphs(controller)

        if i % 1000:
            scores = []
            for genera_id, genera_species in controller.graphs.items():
                for species_id, species_graph in genera_species.items():
                    combined_score = 0
                    for train in training_list:
                        result = species_graph.run(train[0])
                        score = int((1 - (train[1] - result[0][0])) * 100)
                        combined_score += score
                    scores.append(combined_score)
            avg_score = sum(scores) / len(scores)
            print(f'Average score: {avg_score}')


if __name__ == '__main__':
    random.seed(1)
    controller = core.NeatController(1, 20, {'a': 'enter', 'b': 'enter', 'c': 'exit'})
    train(500, controller)
