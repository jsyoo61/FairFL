# %%
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import os

# %%
def exp(n_E = 10, n_A = 8, n_U = 80, E_batchsize = 1024, U_batchsize = 128, I_threshold = 5000, schedule = 'random'):
    E = np.zeros(n_E)
    U = np.zeros(n_U)
    I = 0
    I_hist = []
    I_sum = 0

    while I_sum < I_threshold:
        # 1]. U receive random amount of data
        # print(I_sum, E)
        U += np.random.randint(U_batchsize, size = n_U)
        np.clip(U, 0, U_batchsize, out = U)

        # 2]. Schedule U and E. Distribute data
        U, E = schedule_f[schedule](U, E, n_A, E_batchsize)

        # 3]. Check if E is full. (I occured)
        I = E >= E_batchsize
        I_sum += sum(I)
        I_hist.append(I)

        # 4]. Flush E
        E[E >= E_batchsize] -= E_batchsize
        # plt_status(U, E)
        # plt.show()

    I_hist = np.asarray(I_hist)
    return I_hist

# %% PLot Functions
def results(I_hist):
    # plt_I_hist(I_hist)
    # plt_I_hist_sep(I_hist)
    I_mean = np.mean(np.sum(I_hist, axis = 1))
    T = len(I_hist) # I_hist.shape[0]
    return I_mean, T

def plt_I_sum(I_hist):
    plt.figure()
    plt.bar(range(I_hist.shape[-1]), np.sum(I_hist, axis = 0))

def plt_I_hist(I_hist):
    plt.figure()
    plt.plot(np.sum(I_hist, axis = 1))

def plt_I_hist_sep(I_hist, title = ''):
    plt.figure()
    I_hist = np.cumsum(I_hist, axis = 0)
    plt.plot(I_hist)
    plt.title(title)
    plt.legend(['Edge: {}'.format(i) for i in range(I_hist.shape[-1])])

def plt_status(U, E):
    plt.figure()
    plt.bar(range(len(E)), E)
    plt.figure()
    plt.bar(range(len(U)), U)

def plt_factorial_exp(x, param_grid, I_mean_list, T_list):
    # Sort order with respect to axis "x"
    shape = tuple([len(param_grid[param]) for param in sorted(param_grid.keys())])
    x_axis = sorted(param_grid.keys()).index(x)
    n_x_tick = len(param_grid[x])
    axis_order = list(range(len(param_grid)))
    axis_order.remove(x_axis)
    axis_order.append(x_axis)
    # print(shape, x_axis, n_x_tick, axis_order)

    # Sort results. Resulting shape equals: (-1, n_x_tick)
    params = np.array(ParameterGrid(param_grid)).reshape(shape)
    params = params.transpose(axis_order).reshape(-1, n_x_tick)
    I_mean_list = np.asarray(I_mean_list).reshape(shape)
    I_mean_list = I_mean_list.transpose(axis_order).reshape(-1, n_x_tick)
    T_list = np.asarray(T_list).reshape(shape)
    T_list = T_list.transpose(axis_order).reshape(-1, n_x_tick)


    legend = params[:,0]
    for param in legend:
        del param[x]
    legend

    fig = plt.figure(figsize = (9,6))
    # plt.figure()
    plt.plot(param_grid[x], I_mean_list.T, 'x-')
    plt.legend(legend)
    plt.xlabel(x)
    plt.ylabel('E[I]')
    # savefig(fig)

    fig = plt.figure(figsize = (9,6))
    # plt.figure()
    plt.plot(param_grid[x], T_list.T, 'x-')
    plt.legend(legend)
    plt.xlabel(x)
    plt.ylabel('T')
    # savefig(fig)

def savefig(fig):
    fig_savedir = str(len(os.listdir(FIG_DIR)) + 1) + '.png'
    fig_savedir = os.path.join(FIG_DIR, fig_savedir)
    fig.savefig(fig_savedir)

# %% Schedule Functions
def schedule_random(U, E, n_A, E_batchsize):
    n_match = len(E) * n_A
    # 1] Enough U to match
    if n_match <= len(U):
        # 1. Random index & random matching
        random_i = np.arange(len(U))
        np.random.shuffle(random_i)
        random_i = random_i[:n_match]

        # 2. Deliver packet
        packet = np.sum(U[random_i].reshape(len(E), n_A), axis = 1)
        E += packet

        # 3. Flush device U
        U[random_i] = 0

    # 2] Not enough U to match
    else:
        n_match = len(U)
        # 1. Random index & random matching
        random_i = np.arange(n_match)
        np.random.shuffle(random_i)

        # 2. Allocate Antennas
        allocated_A_ = [[] for i in range(len(E))] # Allocation in progress
        allocated_A = [] # Allocation complete
        n_left_E = len(allocated_A_)
        for i in random_i:
            selected_E_i = np.random.randint(n_left_E)
            allocated_A_[selected_E_i].append(i)
            # Maximum number of A full
            if len(allocated_A_[selected_E_i]) == n_A:
                assert len(allocated_A_[selected_E_i]) <= n_A
                allocated_A.append(allocated_A_[selected_E_i])
                del allocated_A_[selected_E_i]
                n_left_E = len(allocated_A_)

        allocated_A.extend(allocated_A_)
        np.random.shuffle(allocated_A)

        # 3. Deliver packet
        packet = []
        for i in allocated_A:
            packet.append(sum(U[i]))
        packet = np.asarray(packet)
        E += packet

        # 4. Flush device U
        U[:] = 0

    return U, E

def schedule_greedy_p(U, E, n_A, E_batchsize):
    n_match = len(E) * n_A
    # 1] Enough U to match
    if n_match <= len(U):
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

    # 2] Not enough U to match
    else:
        n_match = len(U)
        # 1. Compute vacancies of E
        E_vacancy = E_batchsize - E
        E_vacancy[E_vacancy < 0] = 0
        sorted_E_i = sorted(range(len(E)), key = lambda i: E_vacancy[i])

        # 2. Index with decreasing U values
        sorted_U = sorted(U, reverse = True)

        # 3. Allocate A
        # Allocate at least 1 antenna
        n_allocated_A = np.zeros(len(E), dtype = int)
        if n_match < len(E):
            n_allocated_A[:n_match] = 1
            n_full_A = 0
        else:
            n_allocated_A[:] = 1
            n_full_A = (n_match - len(E)) // (n_A - 1)
            # Allocate antennas greedily
            n_allocated_A[:n_full_A] += (n_A - 1)
            n_allocated_A[n_full_A] += (n_match - len(E)) % (n_A - 1)

        # 4. Deliver packet
        packet = []
        start = 0
        for A in n_allocated_A:
            if A == 0:
                packet.append(0)
            else:
                end = start + A
                packet.append(sum(sorted_U[start:end]))
                start = end
        packet = np.asarray(packet)

        E[sorted_E_i] += packet

        # 5. Flush device U
        U[:] = 0

    return U, E

def schedule_greedy_n(U, E, n_A, E_batchsize):
    n_match = len(E) * n_A
    # 1] Enough U to match
    if n_match <= len(U):
        # 1. Compute vacancies of E
        E_vacancy = E_batchsize - E
        E_vacancy[E_vacancy < 0] = 0
        sorted_E_i = sorted(range(len(E)), key = lambda i: E_vacancy[i], reverse = True)

        # 2. Index with decreasing U values
        sorted_U_i = sorted(range(len(U)), key = lambda i: U[i], reverse = True)
        sorted_U_i = sorted_U_i[:n_match]

        # 3. Deliver packet
        packet = np.sum(U[sorted_U_i].reshape(len(E), n_A), axis = 1)
        E[sorted_E_i] += packet

        # 4. Flush device U
        U[sorted_U_i] = 0

    # 2] Not enough U to match
    else:
        n_match = len(U)
        # 1. Compute vacancies of E
        E_vacancy = E_batchsize - E
        E_vacancy[E_vacancy < 0] = 0
        sorted_E_i = sorted(range(len(E)), key = lambda i: E_vacancy[i], reverse = True)

        # 2. Index with decreasing U values
        sorted_U = sorted(U, reverse = True)

        # 3. Allocate A
        # Allocate at least 1 antenna
        n_allocated_A = np.zeros(len(E), dtype = int)
        if n_match < len(E):
            n_allocated_A[:n_match] = 1
            n_full_A = 0
        else:
            n_allocated_A[:] = 1
            n_full_A = (n_match - len(E)) // (n_A - 1)
            # Allocate antennas greedily
            n_allocated_A[:n_full_A] += (n_A - 1)
            n_allocated_A[n_full_A] += (n_match - len(E)) % (n_A - 1)

        # 4. Deliver packet
        packet = []
        start = 0
        for A in n_allocated_A:
            if A == 0:
                packet.append(0)
            else:
                end = start + A
                packet.append(sum(sorted_U[start:end]))
                start = end
        packet = np.asarray(packet)

        E[sorted_E_i] += packet

        # 5. Flush device U
        U[:] = 0

    return U, E

def schedule_proposed(U, E, n_A, E_batchsize):
    n_match = len(E) * n_A
    # 1] Enough U to match
    if n_match <= len(U):
        best_U_i = sorted(range(len(U)), key = lambda i: U[i], reverse = True)[:n_match]
        best_U = U[best_U_i]
    else:
        n_match = len(U)
        best_U_i = sorted(range(len(U)), key = lambda i: U[i], reverse = True)
        best_U = U[best_U_i]

    # 1. Create variables (non static variables)
    I = cp.Variable(len(E), boolean = True)
    x = cp.Variable((n_match, len(E)), boolean = True)
    # print('variables created, ',end='')

    # 2. Define Constraints
    constraints = [
    cp.sum(cp.multiply(x, np.broadcast_to(best_U, (len(E), n_match)).T), axis =  0) >= cp.multiply((E_batchsize - E), I),
    cp.sum(x, axis = 1) <= 1,
    cp.sum(x, axis = 0) <= n_A,
    cp.sum(x) == min(len(E) * n_A, n_match)
    ]

    # 3. Create Objective, merge into problem. then solve
    # print('problem ', end='')
    obj = cp.Maximize(sum(I))
    prob = cp.Problem(obj, constraints)
    # prob.solve(solver = cp.CPLEX, verbose=True)
    prob.solve(solver = cp.CPLEX)
    # prob.solve(solver = cp.GLPK_MI,verbose=True)
    # print('solved')
# x.value.sum(axis=1)
# x.value.sum(axis=0)
# x.value
# I.value
# E
# len(U)
    # 4. Deliver packet using optimized result
    packet = np.sum(x.value * np.broadcast_to(best_U, (len(E), n_match)).T, axis = 0)
    E += packet
    flushed_U_i = x.value.sum(axis = 1).astype(bool)
    assert np.logical_or(flushed_U_i == 0, flushed_U_i == 1).all(), 'more than one connection from a single device occured,\n%s'%(flushed_U_i)

    # 5. Flush device U
    # U[flushed_U_i] = 0
    U[best_U_i] = 0

    return U, E
# %%
schedule_f = {
'random': schedule_random,
'greedy_p': schedule_greedy_p,
'greedy_n': schedule_greedy_n,
'proposed': schedule_proposed,
}

from sklearn.model_selection import ParameterGrid
static_param = dict(
E_batchsize = 1024,
U_batchsize = 128,
I_threshold = 100,
# schedule = 'random'
)
param_grid = dict(
# E_batchsize = [1024],
# U_batchsize = [128],
# I_threshold = [5000],
# n_A = [2,4,8,16],
n_A = [8],
# n_A = np.arange(4, 17, 4),
n_E = [8],
# n_U = [16,32],
n_U = np.arange(8, 256, 32),
# n_U = [20],
schedule = ['proposed', 'greedy_p', 'greedy_n']
)

# %%
I_mean_list = []
T_list = []
for grid in ParameterGrid(param_grid):
    print(grid)
    grid.update(static_param)
    I_hist = exp(**grid)
    plt_I_hist_sep(I_hist, title = grid)
    I_mean, T = results(I_hist)

    I_mean_list.append(I_mean)
    T_list.append(T)

# %%
x = 'n_U'
plt_factorial_exp(x, param_grid, I_mean_list, T_list)

for x, y in zip(ParameterGrid(param_grid), I_mean_list):
    print(x,y)

I_mean_list


# %%
n_E = 4
n_A = 4
n_U = 32
E_batchsize = 1024
U_batchsize = 128
I_threshold = 5000
schedule_f = {
'random': schedule_random,
'greedy_p': schedule_greedy_p,
'greedy_n': schedule_greedy_n,
'proposed': schedule_proposed,
}
schedule = 'proposed'
