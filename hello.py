# -*- coding: utf8 -*-

"""Different ways to print "Hello world!\n" in Python.

Some technique requires modules not in the standard library (numpy) that if they
aren't installed they are "emulated".

I know that some modules are imported several times, it's just to do that all
the code of each Hello world is within the commented lines # .. #

Tested with Python 3.6.0 | Anaconda custom (64-bit) | on Ubuntu 16.04 LTS 
  
-------------------------------------WARNING------------------------------------
Make sure you do not have important files that reside in the directory where
this source is executed that have the name starting with "127.0.0.1:8" because
will be deleted.
If it was too late: <<< RTFM >>>
"""

################################################################################


""" Classic Hello World """

print("Hello world!")


################################################################################


""" Easter Egg """

import __hello__


################################################################################


""" Hello world made by directly accessing the standard output. """

import sys

sys.stdout.write("Hello world!\n")


################################################################################


""" Dynamic source code execution """

exec('print("Hello world!")')


################################################################################


"""Python 2 supports the rot-13 encoding by default.
Info: https://en.wikipedia.org/wiki/ROT13
"""

import os

os.system("""python2 -c '# -*- coding: rot13 -*- \ncevag(h"Uryyb jbeyq!")'""")


################################################################################


"""With a system call you run to a new python interpreter the instruction
print('Hello world!').
"""

import os

os.system("python3 -c\"print(\\\"Hello world!\\\")\"")


################################################################################


"""Remember Hello world on CUDA architectures: https://www.pdc.kth.se/resources/computers/historical-computers/zorn/how-to/how-to-compile-and-run-a-simple-cuda-hello-world

The hello array contains ascii codes of 'Hello '.
The array world contains the difference between the N-th code of the hello and
the N-th ascii code of 'world!'.

To get string codes 'world!' just sum (not concatenate) the array hello and
world.
"""

try:
    import numpy as np
except ModuleNotFoundError: # Please download anaconda, has many beautiful games
    from itertools import chain
    class np:
        """ Emulator of basic functions of a Numpy array """
        def __init__(self, arr):
            self.arr = arr
            
        def __add__(self, other):
            return np([a + b for a,b in zip(self.arr, other.arr)])
        
        def __iter__(self):
            return iter(self.arr)
            
        def array(arr):
            return np(arr)
            
        def concatenate(elems):
            return list(chain(*elems))


hello = np.array([ 72, 101, 108, 108, 111,  32])
world = np.array([ 47,  10,   6,   0, -11,   1])

for c in map(chr, np.concatenate((hello, hello + world))):
    print(c, end='')

print() # Newline...


################################################################################


"""A Spam metaclass is created where the __call__ method is overwritten, then
the Egg metaclass is instantiated by the type of Spam metaclass, which is type.

Next, the world instance is created from the Egg metaclass, equivalent to:
    
    >>> world = type(Egg).__call__(Egg, 'world', (), {...})

But since the Egg metaclass is an instance of the Spam metaclass, type (Egg)
returns the Spam metaclass, or rather:

    >>> world = Spam.__call__(Egg, 'world', (), {...})
    
Then the Hello world is printed, and since the __call__ method explicitly does
not return anything the variable world is equal to None.
"""

class Spam(type):
    def __call__(self, name, bases, namespace):
        print("Hello ", name, '!', sep='')


class Egg(metaclass=Spam):
    pass


class world(metaclass=Egg):
    pass


################################################################################


"""The variable code contains the Hello world in Brainfuck: https://docs.google.com/document/d/1M51AYmDR1Q9UBsoTrGysvuzar2_Hx69Hz14tsQXWV6M/mobilebasic
The remaining code is simply an interpreter that interprets the code inside the
code variable.
"""

code = """++++++++[>++++[>++>+++>+++>+<<<<-]
          >+>+>->>+[<]<-]>>.>---.+++++++..+++.
          >>.<+++++++++++++++++++++++++++++++.
          <.+++.------.--------.>>+.>++."""
#code = """++++++++[>++++[>++>+++>+++>+<<<<-]  
#          >+>+>->>+[<]<-]>>.>---.+++++++..+++.
#          >>.<-.<.+++.------.--------.>>+.>++.""" # I wanted the tiny W...
index = 0
pointer = 0
mem = [0] * 65536
while index < len(code):
    loop = 0
    if code[index] == '>':
        pointer = 0 if pointer == 65535 else pointer + 1
    elif code[index] == '<':
        pointer = 65535 if pointer == 0 else pointer - 1
    elif code[index] == '-':
        mem[pointer] -= 1
    elif code[index] == '+':
        mem[pointer] += 1
    elif code[index] == '.':
        print(chr(mem[pointer]), end='')
    elif code[index] == ',':
        mem[pointer] = ord(input()[0])
    elif code[index] == '[':
        if mem[pointer] == 0:
            index += 1
            while loop > 0 or code[index] != ']':
                if code[index] == '[':
                    loop += 1
                if code[index] == ']':
                    loop -= 1
                index += 1
    elif code[index] == ']':
        if mem[pointer] != 0:
            index -= 1
            while loop > 0 or code[index] != '[':
                if code[index] == ']':
                    loop += 1
                if code[index] == '[':
                    loop -= 1
                index -= 1
            index -= 1
    index += 1


################################################################################


""" 128 sockets are instantiated that serve as listening servers on the ports of
localhost from 8000 to 8127, each port map the ascii character with index equal
to the last three digits of the port.
Other 128 hosts are instantiated, which, by sending server signals, allow the
output of the respective mapped character.
The signals are sent in order to print Hello World.
"""

import socket
import threading
from time import sleep

servers = [socket.socket(socket.AF_UNIX) for _ in range(128)]
clients = [socket.socket(socket.AF_UNIX) for _ in range(128)]

hello = (ord(c) for c in "Hello world!\n") # Hello world's ascii codes


def connection(sock):
    """ Accept client requests and based on data received or print ascii
    character corresponding to the last three digits of the socket's door
    or the server is closed and the thread ends.
    """
    c, addr = sock.accept()
    while True:
        data = c.recv(1024)
        if data == b' ':
            break
        print(chr(int(sock.getsockname()[-3:])), end='')
    sock.close()
    
def send_signal(port):
    """ Sends a b'noise' signal to the specified server, sleep serves the threads
    so that it is triggered in the correct order.  If the value drops to .0001
    the likelihood that characters will be printed in disorder is appreciable
    on average 10 attempts, try to believe.
    """
    clients[port].send(b'noise')
    sleep(0.01) 


for i, (client, server) in enumerate(zip(clients, servers)):
    """The clients and the servers are started, and each server creates a thread
    calling the connection function by passing the server itself.
    """
    address = '127.0.0.1:{}'.format(8000 + i)
    
    server.bind(address)
    server.listen(5)
    client.connect(address)
    
    t = threading.Thread(target=connection, args=(server,))
    t.start()


for code in hello:  # Here happens the Hello World
    send_signal(code)

for client in clients: # Sends shutdown signal to servers and closes clients
    client.send(b' ')
    client.close()


# These rows only serve to delete files that are created by sockets
from glob import glob
import os

for f in glob("127.0.0.1:8*"):
    os.remove(f)


################################################################################


""" Hello world using a small neural network """

try:
    import numpy as np
except Exception as e:
    raise e # TODO

def nonlin(x, deriv=False):
    if deriv is True:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# Inputs
x = np.array([[1, 0, 0, 1, 0, 0, 0],
              [1, 1, 0, 0, 1, 0, 1],
              [1, 1, 0, 1, 1, 0, 0],
              [1, 1, 0, 1, 1, 0, 0],
              [1, 1, 0, 1, 1, 1, 1],
              [0, 1, 0, 0, 0, 0, 0],
              [1, 1, 1, 0, 1, 1, 1],
              [1, 1, 0, 1, 1, 1, 1],
              [1, 1, 1, 0, 0, 1, 0],
              [1, 1, 0, 1, 1, 0, 0],
              [1, 1, 0, 0, 1, 0, 0],
              [0, 1, 0, 0, 0, 0, 1],
              [0, 0, 0, 1, 0, 1, 0]])

# Expected outputs
y = np.array([[ord(c) / 150] for c in 'Hello world!\n'])

# Synapsis
syn0 = 2 * np.random.random((7, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1

# Training
for j in range(2500):
    # Layers
    l0 = x
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    
    # Backpropagation
    l2_error = y - l2

    # Calculus of deltas
    l2_delta = l2_error * nonlin(l2, deriv=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1, deriv=True)
    
    # Synapsis update
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)


for c in l2 * 150:
    print(chr(int(round(c[0]))), end='')


################################################################################


""" Hello World using linear algebra and polynomial curve fitting https://www.reddit.com/r/ProgrammerHumor/comments/8ehlev/hello_world_using_linear_algebra_and_polynomial/ """

def f(x):
    return int(round(
        72.0
        - 5601.5239800168910000 * x
        + 15839.254309410564000 * x ** 2
        - 17990.084740472780000 * x ** 3
        + 11078.251208553862000 * x ** 4
        - 4157.1945722233930000 * x ** 5
        + 1004.3607769364212000 * x ** 6
        - 159.60952876624610000 * x ** 7
        + 16.592823896982345000 * x ** 8
        - 1.0862681759576835000 * x ** 9
        + 0.0406327155961238650 * x ** 10
        - 0.0006620771128961875 * x ** 11
    ))

print(''.join(chr(f(i)) for i in range(12)))


################################################################################


""" Bruteforcing Hello world with genetic algorithms 
Translated from here: https://github.com/frogamic/GeneticHelloWorldjs """

import math
import random
from itertools import count

MUTATE_RATE = 0.01
BREED_RATE = 0.75
POPULATION_SIZE = 1000
TARGET = "Hello world!"

def generate_character():
    return chr(random.randint(32, 126))

def select_parent(elders, total_score):
    selection = random.random() * total_score
    sum_ = 0
    for e in elders:
        sum_ += check_fitness(e) # TODO we can use a class that save the score, without computing each time
        if selection <= sum_:
            return e

def generate_population():
    return [''.join(generate_character() for char in TARGET)
            for individual in range(POPULATION_SIZE)]

def check_fitness(x):
    return sum(c1 == c2 for c1, c2 in zip(x, TARGET))

def breed(p1, p2):
    c = ''
    for i, _ in enumerate(TARGET):
        if random.random() < MUTATE_RATE:
            c += generate_character()
        else:
            if random.random() < 0.5:
                c += p1[i]
            else:
                c += p2[i]
    return c


population = generate_population()

for generation in count(1):
    results = sorted(population, key=check_fitness, reverse=True)
    if results[0] != TARGET:
        elders = results[:int(POPULATION_SIZE * (1 - BREED_RATE))]
        population = elders
        total_score = sum(map(check_fitness, population))
        for _ in range(int(POPULATION_SIZE * (1 - BREED_RATE))):
            population.append(breed(select_parent(elders, total_score),
                                    select_parent(elders, total_score)))
    else:
        print(results[0])
        break


################################################################################


""" Hello World! using the discrete Fourier transform https://www.reddit.com/r/ProgrammerHumor/comments/8emakg/i_see_your_hello_world_using_linear_algebra_and/ """

import warnings

from numpy import exp, pi

warnings.filterwarnings('ignore')

coefficients = [
  + 1085.00000000 +   0.00000000j,
  -   31.29422863 +   1.16987298j,
  -  135.00000000 - 136.83201380j,
  +    2.00000000 +  11.00000000j,
  +   20.00000000 - 117.77945491j, 
  -   15.70577137 +   9.83012702j,
  +   99.00000000 +   0.00000000j, 
  -   15.70577137 -   9.83012702j,
  +   20.00000000 + 117.77945491j,
  +    2.00000000 -  11.00000000j,
  -  135.00000000 + 136.83201380j,
  -   31.29422863 -   1.16987298j,
  ]
       
message = ""
for n in range(12):
  message += chr(round(
    sum([
      coefficients[k]*exp(2j*pi*k*n/12)
        for k in range(12)
      ])
    / 12
    ))

print(message)

################################################################################
