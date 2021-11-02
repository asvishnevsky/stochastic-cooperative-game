import numpy as np
from itertools import chain, combinations, product

def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

def V(S, N, Expectations):
    if len(S) == 0:
        return np.zeros(4)
    result = []
    player = 0
    for i, n in enumerate(list(S)):
        player += pow(10, len(list(S)) - i - 1) * n
    eta_S   = list(product([0, 1], repeat = 1))
    eta_N_sub_S = list(product([0, 1], repeat = 1))
    rewards_S = {0: [], 1:[], 2:[], 3: []}
    subgames = [0, 1, 2, 3]
    for our_strategy in eta_S:
        rewards_N_sub_S = {0: [], 1:[], 2:[], 3: []}
        for their_strategy in eta_N_sub_S:
            situation = (our_strategy[0], their_strategy[0])
            for subgame in subgames:
                new_item = Expectations[player][situation].item(subgame)
                rewards_N_sub_S[subgame].append(new_item)
        for subgame in subgames:
            rewards_S[subgame].append ( min(rewards_N_sub_S[subgame]) )
    for subgame in subgames:
        result.append(max(rewards_S[subgame]))
    return np.array(result)

def Sh(i, N, Expectations):
    result = np.zeros(4)
    for coalition in filter(lambda subset: i in subset, all_subsets(N)):
        S = set(coalition)
        s = len(S)
        n = len(N)
        left =  (np.math.factorial(s - 1) * np.math.factorial(n - s) ) / np.math.factorial(n)
        right = V(S, N, Expectations) - V(N - S, N, Expectations)
        result += left * right
    return result

def PI(a1, a2, p1, p2):
    return np.matrix([
                        [(1 - a1) * (1 - a2), (1 - a1) * a2, a1 * (1 - a2), a1 * a2],
                        [(1 - a1) * (1 - a2), (1 - a1) * a2, a1 * (1 - a2), a1 * a2],
                        [(1 - a1) * (1 - a2), (1 - a1) * a2, a1 * (1 - a2), a1 * a2],
                        [(1 - a1) * (1 - a2) * p1 * p2,
                         (1 - a1) * a2 * p1 * p2 + (1 - a1) * (1 - p1 * p2),
                         a1 * (1 - a2) * p1 * p2,
                         a1 * a2 * p1 * p2 + a1 * (1 - p1 * p2)]
                     ])

def check_stability(a1, a2, D12, D21, q, f1, d1, f2, d2, c, pi):
    maxSUM = float('-inf')
    maxSUMstrategy = None
    Expectations = {1: {}, 2: {}, 12: {}}

    for p1, p2 in list(product([0, 1], repeat = 2)):
        K1 = np.array([f1, f1, -c, -c * (1 - p1) + (-c - D12) * p1])
        K2 = np.array([f2,
                       -c,
                       f2,
                       -D21 * p1 * p2 - (c + d2) * p1 * (1 - p2) - (c + d2 + D21) * (1 - p1) * p2 - (c + d2) * (
                                   1 - p1) * (1 - p2)])
        E = np.identity(4)
        A = E - (1 - q) * PI(a1, a2, p1, p1)
        invertedA = np.linalg.inv(A)
        E1 = invertedA.dot(K1)
        Expectations[1][(p1, p2)] = E1
        E2 = invertedA.dot(K2)
        Expectations[2][(p1, p2)] = E1
        SUM = (E1 + E2)
        Expectations[12][(p1, p2)] = SUM
        # print('p1 = {} p2 = {} SUM = {}'.format(p1, p2, SUM.dot(pi).item()))
        if maxSUM < SUM.dot(pi).item():
            maxSUM = SUM.dot(pi).item()
            maxSUMstrategy = (p1, p2)
    # print('Max E{} = {}'.format(maxSUMstrategy, maxSUM))
    first = set([1])
    second = set([2])
    together = set([1, 2])
    V1 = V(first, together, Expectations)
    V2 = V(second, together, Expectations)
    V12 = V(together, together, Expectations)
    # print('V1 = {}'.format(V1))
    # print('V2 = {}'.format(V2))
    # print('V12 = {}'.format(V12))
    VV1 = V1.dot(pi)
    VV2 = V2.dot(pi)
    VV12 = V12.dot(pi)
    # print(VV1, VV2, VV12)
    Sh1 = Sh(1, together, Expectations)
    # print('Sh1 = {}'.format(Sh1))
    Sh2 = Sh(2, together, Expectations)
    # print('Sh2 = {}'.format(Sh2))
    Sh1_whole_game = Sh1.dot(pi)
    Sh2_whole_game = Sh2.dot(pi)
    # print('Sh1_whole_game = {:f} Sh2_whole_game = {}'.format(Sh1_whole_game, Sh2_whole_game))
    E = np.identity(4)
    p1, p2 = maxSUMstrategy
    A = E - (1 - q) * PI(a1, a2, p1, p2)
    beta1 = A.dot(Sh1)
    beta2 = A.dot(Sh2)
    # print('beta1 = {}, beta2 = {}'.format(beta1, beta2))
    if all((beta1 > 0).tolist()[0]) and all((beta2 > 0).tolist()[0]):
        return True
    else:
        return False

def main():
    print ('Stochastic game solver')
    a1 = 0.01
    a2 = 0.01
    D12 = 10
    D21 = 10
    q = 0.01
    f1 = 100000
    f2 = 100000
    d1 = 100000
    d2 =  50
    c = 0.0001
    pi = (0.25, 0.25, 0.25, 0.25)

    print('a1 = {} a2 = {} d1 = {} d2 = {} c = {} D21 = {} f1 = {} f2 = {} => {}'.format(
        a1, a2, d1, d2, c, D21, f1, f2, check_stability(a1, a2, D12, D21, q, f1, d1, f2, d2, c, pi)))


    for a1_ in range(-2, 0):
        a1 = pow(10, a1_)
        for a2_ in range(-2, 0):
            a2 = pow(10, a2_)
            for d2_ in range(0, 2):
                d2 = pow(10, d2_)
                for d1_ in range(5, 6):
                    d1 = pow(10, d1_)
                    for c_ in range(-4, -2):
                        c = pow(10, c_)
                        for D21_ in range(-3, 2):
                            D21 = pow(10, D21_)
                            for f1_ in range(5, 6):
                                f1 = pow(10, f1_)
                                for f2_ in range(5, 6):
                                    f2 = pow(10, f2_)
                                    stable = check_stability(a1, a2, D12, D21, q, f1, d1, f2, d2, c, pi)
                                    if stable:
                                        print ('a1 = {} a2 = {} d1 = {} d2 = {} c = {} D21 = {} f1 = {} f2 = {} => {}'.format(
                                            a1, a2, d1, d2, c, D21, f1, f2, stable) )


if __name__ == '__main__':
    main()