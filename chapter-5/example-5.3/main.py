# @Author: aravind
# @Date:   2017-02-05T23:53:52+05:30
# @Last modified by:   aravind
# @Last modified time: 2017-02-05T23:53:52+05:30


"""
Monte Carlo control with Exploring starts for BlackJack
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import sys

CARDS_LIST = ['A'] + map(str, range(2, 11)) + map(str, [10]*3)
ACTIONS = ['H', 'S']
init = True
MAX_ITER = 10000#*50
iter = 0
G = {}
Q = {}
P = {}
CARD_INT = {}
for i, v in enumerate(CARDS_LIST[:10]):
    CARD_INT[v] = i + 1

class Params:
    pass

class State(Params):
    def __init__(self, h = (12,1,False)):
        self.sum, self.d_card, self.usable = h[0], CARDS_LIST[h[1]-1], bool(h[2])

    def __str__(self):
        return ' '.join(map(str, [self.sum, self.d_card, self.usable]))

    def __hash__(self):
        return (self.sum, CARD_INT[self.d_card], int(self.usable))

def val(cards):
    c = sorted(cards[2:])
    s = cards[0]
    l = len(c)
    for i in range(l):
        cc = c[i]
        if cc == 'A':
            break
        else:
            s += int(cc)
    u = cards[1]
    if not l == 0:
        if (s+11) > 21:
            u = False
        else:
            u = cards[1]

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
    s.d_card = str(d_cards[0]) # d_cards[0] - dealer's visible card

    return s

def get_action(s):
    # Some deterministic policy
    h = s.__hash__()
    return P[h]

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
        G[ss][a].append(g)

def compute_q_values():
    for k, v in G.iteritems():
        Q[k]['H'] = np.mean(v['H'])
        Q[k]['S'] = np.mean(v['S'])

def update_policy(tr):
    for (s, a) in tr:
        h = s.__hash__()
        P[h] = max(Q[h])

def plot_q_values(Q, u):
    X, Y = np.meshgrid(range(1, 11), range(12, 22))
    arr = np.zeros((10, 10))
    for k, v in Q.iteritems():
        if k[2] == u:
            arr[k[0]-12, k[1]-1] = ACTIONS.index(max(v))
    Z = np.reshape(arr, (10, 10))
    fig = plt.figure('Usable ace = ' + str(u))
    plt.scatter(X, Y, c=arr)

m = MAX_ITER/100
print('Computing..')

STATE_HASHES = [ (i,j,k) for i in range(12, 22) for j in range(1, 11) for k in range(2) ]

for s in STATE_HASHES:
    Q[s] = {'H': [], 'S': []}
    P[s] = ACTIONS[1] if (s[0] > 19) else ACTIONS[0]
    G[s] = {'H': [], 'S': []}

while True:
    idx = np.random.randint(len(STATE_HASHES))
    h = STATE_HASHES[idx]
    s = State(h) # Exploring start
    a = ACTIONS[np.random.randint(2)]
    p_cards = [h[0], h[2]]
    d_cards = [h[1], h[2]]

    # Generating an episode
    r = None
    tr = [(s, a)]
    while True:
        r = submit_action(s, a)
        if r is not None:
            break
        s = get_state(p_cards, d_cards)
        a = get_action(s)
        tr.append((s, a))

    print map(lambda x: (x[0].__hash__(), x[1]) ,tr), r
    push_tr(tr, r)
    compute_q_values()
    update_policy(tr)
    iter = iter + 1

    if not iter % m:
        print('Progress: ' + str(iter*100./MAX_ITER) + '%')
    if iter >= MAX_ITER:
        break

print('Done')
# print('Plotting..')
# plot_q_values(Q, 0)
# plot_q_values(Q, 1)
# plt.show()
