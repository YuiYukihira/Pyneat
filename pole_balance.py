import core
import math
import curses
from time import sleep
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Pole:
    def __init__(self, start_angle, mass):
        self.angle = start_angle
        self.mass = mass
        self.vel = 0

    def update(self, dt):
        self.angle += self.vel
        self.vel = self.mass*9.81*math.sin(math.radians(self.angle))*dt
        return self.angle, self.vel

    def wiggle(self, amount):
        self.vel += amount

class PoleTester:
    def __init__(self, dt, start_angle, mass):
        self.screen = curses.initscr()
        self.current_pole_win = curses.newwin(8, 40, 0, 0)
        self.high_scores_win = curses.newwin(8, 30, 0, 41)
        self.genome_win = curses.newwin(20, 40, 8, 0)
        self.high_genome_win = curses.newwin(20, 40, 8, 41)
        self.controller = core.NeatController(10, 10, {'a': 'enter', 'v': 'enter', 'w': 'exit'})
        self.dt = dt
        self.start_angle = start_angle
        self.mass = mass
        self.generation_counter = 0
        self.high_scores = [[0, (0, 0), 0]]*5

    def display_high_scores(self):
        self.high_scores_win.refresh()
        self.high_scores_win.addstr(0, 0, 'HIGH SCORES:')
        for i in range(0, 5):
            self.high_scores_win.addstr(i+1, 0, f'{i+1}: {self.high_scores[i][0]} by {self.high_scores[i][1]} in gen: {self.high_scores[i][2]}')

    def display_genome(self, screen, genome, gid, generation):
        screen.refresh()
        screen.erase()
        screen.addstr(0, 0, f'Genome for: {gid} in {generation}')
        screen.addstr(1, 0, f'{genome}')

    def test_once(self):
        score = 0
        test_pole = Pole(self.start_angle, self.mass)
        while abs(test_pole.angle) < 90:
            self.current_pole_win.refresh()
            self.current_pole_win.erase()
            pole_state = test_pole.update(self.dt)
            neat_result = self.controller.run({'a': pole_state[0], 'v': pole_state[1]})
            test_pole.wiggle(neat_result[0][0])
            self.current_pole_win.addstr(0, 0, f'Generation: {self.generation_counter//(self.controller.genera_count*self.controller.species_count)}')
            self.current_pole_win.addstr(1, 0, f'NN: {self.controller.current}')
            self.current_pole_win.addstr(2, 4, f'Pole state:')
            self.current_pole_win.addstr(3, 8, f'Angle: {pole_state[0]}')
            self.current_pole_win.addstr(4, 8, f'Vel:   {pole_state[1]}')
            self.current_pole_win.addstr(5, 4, f'Neat result: {neat_result[0][0]}')
            self.current_pole_win.addstr(6, 0, f'SCORE: {score}')
            score += 1
            self.display_genome(self.genome_win, self.controller.graphs[self.controller.current[0]][self.controller.current[1]].genotype, self.controller.current, self.generation_counter)
            if score > 10000:
                break
            scores_copy = self.high_scores.copy()
            for high_score in range(len(scores_copy)):
                if score > scores_copy[high_score][0]:
                    self.high_scores[high_score] = [score, self.controller.current, self.generation_counter//(self.controller.genera_count*self.controller.species_count)]
                    self.display_genome(self.high_genome_win, self.controller.graphs[self.high_scores[0][1][0]][self.high_scores[0][1][1]].genotype, self.controller.current, self.high_scores[0][2])
                    break
            self.display_high_scores()
            #sleep(0.1)
        self.controller.game_over(score)

    def run(self):
        while True:
            try:
                self.test_once()
                self.generation_counter += 1
            except KeyboardInterrupt:
                break

if __name__ == '__main__':
    pole_test = PoleTester(0.1, 1, 1)
    pole_test.run()
