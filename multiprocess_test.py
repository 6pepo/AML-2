import numpy as np
import time as clock
import matplotlib.pyplot as plt
# import RF_Library as RF

import multiprocessing as mp
import os
import random

#
# Function run by worker processes
#

def worker(input, output):
    for func, args in iter(input.get, 'STOP'):
        result = calculate(func, args)
        output.put(result)

#
# Function used to calculate result
#

def calculate(func, args):
    result = func(*args)
    return '{} says that {}{} = {}'.format(mp.current_process().name, func.__name__, args, result)

#
# Functions referenced by tasks
#

def mul(a, b):
    clock.sleep(0.5*random.random())
    return a * b

def plus(a, b):
    clock.sleep(0.5*random.random())
    return a + b

#
#
#

def test():
    NUMBER_OF_PROCESSES = 4
    print(__name__)
    TASKS1 = [(mul, (i, 7)) for i in range(20)]
    TASKS2 = [(plus, (i, 8)) for i in range(10)]

    # Create queues
    task_queue = mp.Queue()
    done_queue = mp.Queue()

    # Submit tasks
    for task in TASKS1:
        task_queue.put(task)

    # Start worker processes
    for i in range(NUMBER_OF_PROCESSES):
        mp.Process(target=worker, args=(task_queue, done_queue)).start()

    # Get and print results
    print('Unordered results:')
    for i in range(len(TASKS1)):
        print('\t', done_queue.get())

    # Add more tasks using `put()`
    for task in TASKS2:
        task_queue.put(task)

    # Get and print some more results
    for i in range(len(TASKS2)):
        print('\t', done_queue.get())

    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put('STOP')

print(__name__)
if __name__ == '__main__':
    mp.freeze_support()

    tot_cpu = mp.cpu_count()
    print("Total number of cpu: {}".format(tot_cpu))
    # available_cpu = os.cpu_count()
    # print("Available number of cpu: {}".format(available_cpu))
    test()


    plt.show()
