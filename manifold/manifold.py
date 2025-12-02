import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def read_source(source):
    states = np.array([
        channel.read()
        for channel in source])

    return states


class State():
    def __init__(self, state):
        self.state = state

    def read(self):
        return self.state


def states_array(self, states):
    return [
        State(state)
        for state in states]


class Node():
    def __init__(self, combine, inputs, sensitivity=0.1, adaptation=0.1):
        self.memory = 0
        self.sensitivity = sensitivity
        self.adaptation = adaptation
        self.combine = combine

        # may need to split this apart
        self.inputs = inputs


    def sample(self):
        input_vector = read_source(
            self.inputs)
        input = self.combine(input_vector)
        scaled = input * self.sensitivity
        error = scaled - self.memory
        self.memory += error * self.adaptation

        return error
        

    def read(self):
        return self.level - self.memory


def add(states):
    return np.sum(states)


def plot(x, ys, labels=None, xlabel='time', ylabel='levels', title='plot'):
    if labels is None:
        labels = range(len(ys))

    for y, label in zip(ys, labels):
        plt.plot(x, y, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    plt.savefig(f'out/{title}.png')


def test_node():
    off = [State(0)]
    on = [State(1)]
    history = []
    interval = 111

    # TODO: make a timeline

    node = Node(add, off)
    history += [
        (node.sample(), node.memory)
        for time in range(interval)]

    node.inputs = on
    history += [
        (node.sample(), node.memory)
        for time in range(interval)]

    node.inputs = off
    history += [
        (node.sample(), node.memory)
        for time in range(interval)]

    threads = list(zip(*history))

    plot(
        range(len(history)),
        threads,
        labels=['error', 'memory'])

    import ipdb; ipdb.set_trace()


def test_homeostat():
    # TODO: implement homeostat
    pass


def test_multiple_inputs():

    initial_states = np.array([
        1., 0., 0., 0., 0.])

    states = states_array(initial_states)
    node = Node(states)


if __name__ == '__main__':
    test_node()
