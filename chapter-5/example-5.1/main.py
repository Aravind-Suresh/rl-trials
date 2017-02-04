# @Author: aravind
# @Date:   2017-02-04T19:46:23+05:30
# @Last modified by:   aravind
# @Last modified time: 2017-02-04T19:46:23+05:30


"""
Monte Carlo estimation of value functions for BlackJack for a given policy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import sys

CARDS_LIST = ['A'] + map(str, range(2, 11)) + map(str, [10]*3)
ACTIONS = ['H', 'S']
tr = []
init = True
MAX_ITER = 10000*50
iter = 0
G = {}
V = {}
CARD_INT = {}
for i, v in enumerate(CARDS_LIST[:10]):
    CARD_INT[v] = i + 1

class Params:
    pass

class State(Params):
    def __str__(self):
        return ' '.join(map(str, [self.sum, self.d_card, self.usable]))

    def __hash__(self):
        return (self.sum, CARD_INT[self.d_card], int(self.usable))

def val(cards):
    c = sorted(cards)
    s = 0
    l = len(c)
    for i in range(l):
        cc = c[i]
        if cc == 'A':
            break
        else:
            s += int(cc)
    u = False
    if (s+11) > 21:
        u = False
    else:
        u = True

    if u:
        s += (11 + (l-1-i))
    else:
        s += (l-i)

    return s, u

def deal_card():
    # Infinite deck
    idx = np.random.randint(len(CARDS_LIST))
    return CARDS_LIST[idx]

def get_state(p_cards, d_cards):
    s = State()

    s_p, u_p = val(p_cards)

    s.usable = u_p
    s.sum = s_p
    s.d_card = d_cards[0] # d_cards[0] - dealer's visible card

    return s

def get_action(s):
    # Some deterministic policy
    if s.sum <= 19:
        return ACTIONS[0]
    else:
        return ACTIONS[1]

def submit_action(s, a):
    # Feedback from the environment
    # Dealer follows a deterministic policy
    if a == 'H':
        p_cards.append(deal_card())
        s_p = val(p_cards)
        if s_p > 21:
            return -1

        s_d = val(d_cards)
        if s_d > 21:
            # Dealer busted; Player wins
            return 1
        elif s_d >= 17:
            # Dealer sticks
            return np.sign(s_p - s_d)
        else:
            # Dealer deals. New card is hidden
            d_cards.append(deal_card())
            return None
    else:
        # Player sticks
        s_p, _ = val(p_cards)
        s_d, _ = val(d_cards)
        return np.sign(s_p - s_d)

def push_tr(tr, g):
    # tr - trajectory
    # g - return ( = terminal reward in this case )
    # print 'Updated'
    for (s, a) in tr:
        ss = s.__hash__()
        if not G.has_key(ss):
            G[ss] = []
        G[ss].append(g)

def compute_values():
    for k, v in G.iteritems():
        V[k] = np.mean(v)

def plot_values(V, u):
    X, Y = np.meshgrid(range(1, 11), range(12, 22))
    arr = np.zeros((10, 10))
    for k, v in V.iteritems():
        if k[2] == u:
            arr[k[0]-12, k[1]-1] = v
    Z = np.reshape(arr, (10, 10))
    fig = plt.figure('Usable ace = ' + str(u))
    ax = axes3d.Axes3D(fig)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

m = MAX_ITER/100
print('Computing..')

while True:
    if init:
        p_cards = [ deal_card() for i in range(2) ]
        d_cards = [ deal_card() for i in range(2) ]
        init = False
    s = get_state(p_cards, d_cards)
    a = get_action(s)
    r = submit_action(s, a)
    if not iter % m:
        print('Progress: ' + str(iter*100./MAX_ITER) + '%')
    # print i, s, a, r
    tr.append((s, a))
    if r is not None:
        push_tr(tr, r)
        iter = iter + 1
        tr = []
        init = True
    if iter >= MAX_ITER:
        break
compute_values()

print('Done')
print('Plotting..')
plot_values(V, 0)
plot_values(V, 1)
plt.show()
