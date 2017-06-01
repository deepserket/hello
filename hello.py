# -*- coding: utf8 -*-
""" Diversi modi di stampare "Hello world!\n" in Python.
    Qualche tecnica richiede moduli non presenti nella libreria standard (numpy)
    che se non sono installati vengono '''''''emulati'''''''.
    
    So che alcuni moduli vengono importati più volte, è semplicemente per far
    si che tutto il codice di ogni Hello world stia all'interno delle righe 
    commentate ####...####.
    
    Testato su Python 3.6.0 | Anaconda custom (64-bit) | su Ubuntu 16.04 LTS 
    
    ---------------------------------ATTENZIONE---------------------------------
    Accertarsi di non avere file importanti che risiedono nella directory in cui
    questo sorgente viene eseguito che hanno nome iniziante per "127.0.0.1:8"
    perchè verranno eliminati.
    Nel caso fosse troppo tardi: <<< RTFM >>>
    """

################################################################################


""" Classico Hello World """

print("Hello world!")


################################################################################


""" Easter Egg """

import __hello__


################################################################################


""" Hello world fatto accedendo direttamente allo standard output. """

import sys

sys.stdout.write("Hello world!\n")


################################################################################


""" Esecuzione dinamica di codice sorgente """

exec('print("Hello world!")')


################################################################################


""" A quanto pare Python 2 supporta di default l'encoding rot-13 del sorgente
    info: https://en.wikipedia.org/wiki/ROT13 """

import os

os.system("""python2 -c '# -*- coding: rot13 -*- \ncevag(h"Uryyb jbeyq!")'""")


################################################################################


""" Con una chiamata di sistema si fa eseguire ad un interprete python 
    l'istruzione print('Hello world!'). """

import os

os.system("python3 -c\"print(\\\"Hello world!\\\")\"")


################################################################################


""" Ricorda l'Hello world sulle architetture CUDA: https://www.pdc.kth.se/resources/computers/historical-computers/zorn/how-to/how-to-compile-and-run-a-simple-cuda-hello-world
    L'array hello contiene i codici ascii di 'Hello '.
    L'array world contiene la differenza tra l' N-esimo codice dell'array hello
    e l' N-esimo codice ascii di 'world!'.
    Per ottenere i codici della stringa 'world!' basta sommare (non concatenare)
    gli array hello e world. """

try:
    import numpy as np
except ModuleNotFoundError: # Per favore scarica anaconda, ha tanti bei giochi
    from itertools import chain
    class np:
        """ Emulatore delle funzioni base di un Numpy array"""
        def __init__(self, arr):
            self.arr = arr
            
        def array(arr):
            return np(arr)
            
        def __add__(self, other):
            return np([a + b for a,b in zip(self.arr, other.arr)])
            
        def concatenate(elems):
            return list(chain(*elems))
        
        def __iter__(self):
            return iter(self.arr)


hello = np.array([ 72, 101, 108, 108, 111,  32])
world = np.array([ 47,  10,   6,   0, -11,   1])

for c in map(chr, np.concatenate((hello, hello + world))):
    print(c, end='')

print() # Newline...


################################################################################



""" Viene creata una metaclasse Spam dove viene sovrascritto il metodo __call__
    Poi viene creata la metaclasse Egg che viene istanziata dal tipo della 
    metaclasse Spam, che è type.
    Dopo viene creata l'istanza World dalla metaclasse Egg, equivalente a:
    
world = type(Egg).__call__(Egg, 'world', (), {...})

    Ma visto che la metaclasse Egg è un'istanza della metaclasse Spam type(Egg)
    restituisce la metaclasse Spam, ovvero:

world = Spam.__call__(Egg, 'world', (), {...})

    Quindi viene stampato l'Hello world, e visto che il metodo __call__ 
    esplicitamente non ritorna nulla la variabile World è uguale a None.
"""

class Spam(type):
    def __call__(self, name, bases, namespace):
        print("Hello ", name, '!', sep='')


class Egg(metaclass=Spam):
    pass


class world(metaclass=Egg):
    pass



################################################################################


""" La variabile code contiene l'Hello world in Brainfuck: https://docs.google.com/document/d/1M51AYmDR1Q9UBsoTrGysvuzar2_Hx69Hz14tsQXWV6M/mobilebasic
    Il restante codice è semplicemente un interprete che, appunto, interpreta il
    codice dentro la variabile code. """

code = """++++++++[>++++[>++>+++>+++>+<<<<-]
          >+>+>->>+[<]<-]>>.>---.+++++++..+++.
          >>.<+++++++++++++++++++++++++++++++.
          <.+++.------.--------.>>+.>++."""
#code = """++++++++[>++++[>++>+++>+++>+<<<<-]  
#          >+>+>->>+[<]<-]>>.>---.+++++++..+++.
#          >>.<-.<.+++.------.--------.>>+.>++.""" # Volevo la W minuscola...
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


""" Vengono istanziati 128 socket che fungono da server in ascolto sulle porte
    di localhost dalla 8000 alla 8127, ogni porta mappa il carattere ascii con
    indice pari alle ultime tre cifre.
    Vengono istanziati altrettanti host che, inviando segnali ai server,
    permettono l'output del rispettivo carattere mappato.
    I segnali vengono inviati in ordine tale da stampare Hello World"""

import socket
import threading
from time import sleep

servers = [socket.socket(socket.AF_UNIX) for _ in range(128)]
clients = [socket.socket(socket.AF_UNIX) for _ in range(128)]

hello = (ord(c) for c in "Hello world!\n") # Codici ascii dell'Hello world


def connection(sock):
    """ Accetta le richieste dei client e in base ai dati ricevuti o stampa il 
    carattere ascii corrispondente alle ultime tre cifre della porta del socket
    o il server viene chiuso e il thread termina """
    c, addr = sock.accept()
    while True:
        data = c.recv(1024)
        if data == b' ':
            break
        print(chr(int(sock.getsockname()[-3:])), end='')
    sock.close()
    
def send_signal(port):
    """ Invia un segnale  b'noise' al server indicato, lo sleep serve ai thread
    in modo da venire attivati nell'ordine corretto, se il valore scende a .0001
    la probabilità che i caratteri vengano stampati in disordine è apprezzabile
    in mediamente 10 tentativi, provare per credere """
    clients[port].send(b'noise')
    sleep(0.01) 


for i, (client, server) in enumerate(zip(clients, servers)):
    """ Vengono avviati i client ed i server e per ogni server si crea un thread
    chiamando la funzione connection passandole il server stesso """
    address = '127.0.0.1:{}'.format(8000 + i)
    
    server.bind(address)
    server.listen(5)
    client.connect(address)
    
    t = threading.Thread(target=connection, args=(server,))
    t.start()


for code in hello:  # Qua avviene l'Hello World
    send_signal(code)

for client in clients: # Manda segnale di chiusura ai server e chiude i client
    client.send(b' ')
    client.close()


# Queste righe servono solo ad eliminare i file che vengono creati dai socket
from glob import glob
import os

for f in glob("127.0.0.1:8*"):
    os.remove(f)


################################################################################


""" Hello world utilizzando una piccola rete neurale """
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

# Synapses
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

    # Calculate deltas
    l2_delta = l2_error * nonlin(l2, deriv=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1, deriv=True)
    
    # Update synapses
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)


for l in l2 * 150:
    print(chr(int(round(l[0]))), end='')

