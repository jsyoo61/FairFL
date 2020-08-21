# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
def exp(n_E = 10, n_A = 8, n_U = 80, E_batchsize = 1024, U_batchsize = 128, I_threshold = 5000, schedule = 'random'):
    E = np.zeros(n_E)
    U = np.zeros(n_U)
    I = 0
    I_hist = []
    I_sum = 0
    print(n_A)
    while I_sum < I_threshold:
    # for i in range(200):
        # 1]. U receive random amount of data
        U += np.random.randint(U_batchsize, size = n_U)
        np.clip(U, 0, U_batchsize, out = U)

        # 2]. Schedule U and E. Distribute data
        # U, E = schedule_random(U, E, n_A)
        U, E = schedule_f[schedule](U, E, n_A)

        # 3]. Check if E is full. (I occured)
        I = E >= E_batchsize
        I_sum += sum(I)
        I_hist.append(I)

        # 4]. Flush E
        E[E >= E_batchsize] -= E_batchsize
        # plt_status(U, E)

    I_hist = np.asarray(I_hist)
    return I_hist

# %% PLot Functions
def results(I_hist):
    plt_I_hist(I_hist)
    plt_I_hist_sep(I_hist)
    I_mean = np.mean(np.sum(I_hist, axis = 1))
    T = len(I_mean) # I_mean.shape[0]
    return I_mean, T

def plt_I_hist(I_hist):
    plt.figure()
    plt.plot(np.sum(I_hist, axis = 1))

def plt_I_hist_sep(I_hist):
    plt.figure()
    I_hist = np.cumsum(I_hist, axis = 0)
    plt.figure()
    plt.plot(I_hist)
    plt.legend(['Edge: {}'.format(i) for i in range(n_E)])

def plt_status(U, E):
    plt.figure()
    plt.bar(range(len(E)), E)
    plt.figure()
    plt.bar(range(len(U)), U)

# %% Schedule Functions
def schedule_random(U, E, n_A):
    n_match = len(E) * n_A
    assert n_match <= len(U), 'Number of matches({}) exceeds number of devices({})'.format(n_match, len(U))
    # 1. Random index & random matching
    random_i = np.arange(len(U))
    np.random.shuffle(random_i)
    random_i = random_i[:n_match]

    # 2. Deliver packet
    packet = np.sum(U[random_i].reshape(len(E), n_A), axis = 1)
    E += packet

    # 3. Flush device U
    U[random_i] = 0

    return U, E

def schedule_greedy(U, E, n_A):
    n_match = len(E) * n_A
    assert n_match <= len(U), 'Number of matches({}) exceeds number of devices({})'.format(n_match, len(U))
    # 1. Compute vacancies of E
    E_vacancy = E_batchsize - E
    E_vacancy[E_vacancy < 0] = 0
    sorted_E_i = sorted(range(len(E)), key = lambda i: E_vacancy[i])

    # 2. Index with decreasing U values
    sorted_U_i = sorted(range(len(U)), key = lambda i: U[i], reverse = True)
    sorted_U_i = sorted_U_i[:n_match]

    # 3. Deliver packet
    packet = np.sum(U[sorted_U_i].reshape(len(E), n_A), axis = 1)
    E[sorted_E_i] += packet

    # 4. Flush device U
    U[sorted_U_i] = 0

    return U, E

# %%
help(np.random.randint)
f(U)
U[40:50] = 0
U
x = U[40:50]

x += 10
U

sorted_E_i
E_vacancy
plt.bar(range(10), E)
plt.bar(range(10), E[sorted_E_i])
plt.bar(range(10), E_vacancy)
plt.bar(range(10), E_vacancy[sorted_E_i])
def f(x):
    x += 10
