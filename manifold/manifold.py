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
        self.level = 0
        self.memory = 0
        self.sensitivity = sensitivity
        self.adaptation = adaptation
        self.combine = combine

        # may need to split this apart
        self.inputs = inputs
        # self.membrane = np.zeros((len(inputs), len(inputs)))


    def sample(self):
        input_vector = read_source(
            self.inputs)
        input = self.combine(input_vector)
        scaled = input * self.sensitivity
        learned = self.memory * self.adaptation
        error = scaled - learned
        self.memory = error

        return error
        

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


def plot(x, y, xlabel='time', ylabel='level', title='plot'):
    plt.plot(x, y)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.savefig(f'out/{title}.png')


def test_node():
    off = [State(0)]
    on = [State(1)]
    history = []
    interval = 111

    # TODO: make a timeline

    node = Node(add, off)
    history += [
        node.sample()
        for time in range(interval)]

    node.inputs = on
    history += [
        node.sample()
        for time in range(interval)]

    node.inputs = off
    history += [
        node.sample()
        for time in range(interval)]

    plot(range(len(history)), history)

    import ipdb; ipdb.set_trace()


def test_multiple_inputs():

    initial_states = np.array([
        1., 0., 0., 0., 0.])

    states = states_array(initial_states)
    node = Node(states)


if __name__ == '__main__':
    test_node()
