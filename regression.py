import numpy as np
from sympy import *
import operator
from random import randint, random, shuffle
from copy import deepcopy
import re
from utils.seeds import random_seed
import h5py


def main(number, data_name):
    random_seed(0)

    POP_SIZE = 500
    MAX_GENERATIONS = 80
    TOURNAMENT_SIZE = round(POP_SIZE * 0.03)

    file = open('non_' + data_name + '_' + number + ".txt", 'w')

    with h5py.File(data_name + '.h5', 'r') as f:
        X_train = np.array(f[number]['x_train']).T
        y_label = np.array(f[number]['y_train']).T

    def rand_w():
        return str(np.random.randint(low=-20, high=20))

    def render_prog(node):
        if "children" not in node:
            return node["feature_name"]
        return node["format_str"].format(*[render_prog(c) for c in node["children"]])

    def simp(tree):
        return str(expand(sympify(render_prog(tree)))).replace("*", "@").replace('@@', '**')

    def evaluate(expr, x_data):
        x = x_data
        temp = re.split(' ', expr)
        for n, i_exp in enumerate(temp):
            if '@' in i_exp:
                index = i_exp.find('@')
                tem = list(i_exp)
                tem[index - 1] = str((eval(''.join(i_exp[0:index])) * np.ones((1, x_data.shape[0]))).tolist())
                del tem[0:index - 1]
                temp[n] = ''.join(tem)
        ex = ''.join(temp)
        return expr, eval(ex)

    operations = (
        {"func": operator.add, "arg_count": 2, "format_str": "({} + {})"},
        {"func": operator.sub, "arg_count": 2, "format_str": "({} - {})"},
        {"func": operator.mul, "arg_count": 2, "format_str": "({} * {})"},
        {"func": operator.neg, "arg_count": 1, "format_str": "-({})"},
    )

    X_PCT = 0.7

    def random_prog(depth=0):
        n = 0
        if randint(0, 10) >= depth and n <= 6:
            op = operations[randint(0, len(operations) - 1)]
            n += 1
            return {
                "func": op["func"],
                "children": [random_prog(depth + 1) for _ in range(op["arg_count"])],
                "format_str": op["format_str"],
            }
        else:
            return {"feature_name": 'x'} if random() < X_PCT else {"feature_name": rand_w()}

    def select_random_node(selected, parent, depth):
        if "children" not in selected:
            return parent
        if randint(0, 10) < 2 * depth:
            return selected
        child_count = len(selected["children"])
        return select_random_node(
            selected["children"][randint(0, child_count - 1)],
            selected, depth + 1)

    def do_mutate(selected):
        offspring = deepcopy(selected)
        mutate_point = select_random_node(offspring, None, 0)
        child_count = len(mutate_point["children"])
        mutate_point["children"][randint(0, child_count - 1)] = random_prog(0)
        return offspring

    def do_xover(selected1, selected2):
        offspring = deepcopy(selected1)
        xover_point1 = select_random_node(offspring, None, 0)
        xover_point2 = select_random_node(selected2, None, 0)
        child_count = len(xover_point1["children"])
        xover_point1["children"][randint(0, child_count - 1)] = xover_point2
        return offspring

    def get_random_parent(popu, fitne):
        tournament_members = [
            randint(0, POP_SIZE - 1) for _ in range(TOURNAMENT_SIZE)]

        member_fitness = [(fitne[i], popu[i]) for i in tournament_members]
        return min(member_fitness, key=lambda x: x[0])[1]

    XOVER_PCT = 0.3

    def get_offspring(popula, ftns):
        tempt = random()
        parent1 = get_random_parent(popula, ftns)
        if tempt < XOVER_PCT:
            parent2 = get_random_parent(popula, ftns)
            return do_xover(parent1, parent2)
        elif XOVER_PCT <= tempt < 0.9:
            return do_mutate(parent1)
        else:
            return parent1

    def node_count(x):
        if "children" not in x:
            return 1
        return sum([node_count(c) for c in x["children"]])

    def compute_fitness(fuc, pred):
        m = fuc.count('x')
        if m == 0:
            return float("inf")
        else:
            mse = np.mean(np.square(pred - y_label))
            return mse

    population = [random_prog() for _ in range(POP_SIZE)]
    global_best = float("inf")
    best_prog = ''
    box = {}

    for gen in range(MAX_GENERATIONS):
        fitness = []
        for prog in population:
            func, prediction = evaluate(simp(prog), X_train)
            score = compute_fitness(func, prediction)
            fitness.append(score)

            if score < global_best:
                global_best = score
                best_prog = func

            # ========================================================
            if len(box) < POP_SIZE * 0.05:
                box[score] = prog
            else:
                key_sort = sorted(box)
                if score < key_sort[-1]:
                    box.pop(key_sort[-1])
                    box[score] = prog
        # =======================================================

        print("Generation: {:d}\nBest Score: {:.4f}\nMedian score: {:.4f}\nBest program: {:s}\n" \
              .format(gen, global_best, np.median(np.array(fitness)), best_prog))
        file.write("Generation: {:d}\nBest Score: {:.4f}\nMedian score: {:.4f}\nBest program: {:s}\n\n" \
                   .format(gen, global_best, np.median(np.array(fitness)), best_prog))

        # ========================================================

        lst = []
        lst.extend(box.values())
        population += lst
        shuffle(population)
        population_new = [get_offspring(population, fitness) for _ in range(POP_SIZE)]
        population = population_new + lst

    print("Best score: %f" % global_best)
    file.write("Best score: %f\n" % global_best)
    print("Best program: %s" % best_prog)
    file.write("Best program: %s" % best_prog)
    file.close()


if __name__ == '__main__':
    dataset = ['epsilon_low', 'epsilon_medium', 'epsilon_high']
    num = ['no1', 'no2', 'no3', 'no4', 'no5', 'no6', 'no7', 'no8', 'no9', 'no10']
    for i in dataset:
        for j in num:
            main(j, i)
