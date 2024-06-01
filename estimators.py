import gurobipy as gp
import numpy as np
from algorithms import UCB
import scipy.optimize
import copy

def Baseline1(alg):
    sample_means = np.random.rand(len(alg.sample_means))
    return sample_means

def Baseline2(alg):
    taus = [0] * alg.arm.shape[0]
    num_pulls = {}
    for t, action in enumerate(alg.action_idxs):
        taus[action] = t

    for i in range(len(alg.arm)):
        num_pulls[i] = []

    for t in range(alg.T):
        for key in num_pulls.keys():
            if alg.action_idxs[t] != key:
                if t != 0:
                    num_pulls[key].append(num_pulls[key][ - 1])
                else:
                    num_pulls[key].append(0)
            else:
                if t != 0:
                    num_pulls[key].append(num_pulls[key][ - 1] + 1)
                else:
                    num_pulls[key].append(1)

    optimal_arm = None
    most_pulls = -1
    for i in range(len(alg.arm)):
        if num_pulls[i][alg.T - 1] > most_pulls:
            most_pulls = num_pulls[i][alg.T - 1] 
            optimal_arm = i
    sample_means = []
    for idx, tau in enumerate(taus):
        if idx is not optimal_arm:
            try:
                sample_means.append(UCB.gcb(alg.T, alg.alpha, num_pulls[idx][tau]) - UCB.gcb(alg.T, alg.alpha, num_pulls[optimal_arm][tau]))
            except:
                print(alg.action_idxs)
                print(num_pulls)
        else:
            sample_means.append(0)
    
    return np.asarray(sample_means) * -1

def Baseline2_LP(alg, timelimit=None):
    taus = [-1] * alg.arm.shape[0]
    num_pulls = {}


    for i in range(len(alg.arm)):
        num_pulls[i] = []

    for t in range(alg.T):
        for key in num_pulls.keys():
            if alg.action_idxs[t] != key:
                if t != 0:
                    num_pulls[key].append(num_pulls[key][ - 1])
                else:
                    num_pulls[key].append(0)
            else:
                if t != 0:
                    num_pulls[key].append(num_pulls[key][ - 1] + 1)
                else:
                    num_pulls[key].append(1)

    optimal_arm = None
    most_pulls = -1
    for i in range(len(alg.arm)):
        if num_pulls[i][alg.T - 1] > most_pulls:
            most_pulls = num_pulls[i][alg.T - 1] 
            optimal_arm = i

    past_arm = False
    for t, action in reversed(list(enumerate(alg.action_idxs))):
        if action == optimal_arm:
            past_arm = True
        if past_arm and taus[action] == -1:
            taus[action] = t

    m = gp.Model()
    m.Params.LogToConsole = 0
    if timelimit is not None:
        m.setParam('TimeLimit', timelimit)
    all_vars = {}
    for idx, arm in enumerate(alg.arm):
        all_vars[idx] = m.addVar(name="u_{}".format(idx))

    for idx, tau in enumerate(taus):
        try:
            if idx is not optimal_arm:
                m.addConstr(all_vars[idx] - all_vars[optimal_arm] >= UCB.gcb(alg.T, alg.alpha, num_pulls[optimal_arm][tau-1]) - UCB.gcb(alg.T, alg.alpha, num_pulls[idx][tau-1]))
                m.addConstr(all_vars[idx] - all_vars[optimal_arm] <= UCB.gcb(alg.T, alg.alpha, num_pulls[optimal_arm][tau]) - UCB.gcb(alg.T, alg.alpha, num_pulls[idx][tau]))
        except:
            breakpoint()
            print(num_pulls[optimal_arm][tau-1])
            print(num_pulls[optimal_arm][tau])

            print(num_pulls[idx][tau-1])
            print(num_pulls[idx][tau])

    m.optimize()
    lp_vals = []

    for i in range(len(alg.arm)):
        lp_vals.append(all_vars[i].X)
    
    return lp_vals

def estimate_ucb_means_lp(alg, timelimit=None):
    m = gp.Model()
    m.Params.LogToConsole = 0
    if timelimit is not None:
        m.setParam('TimeLimit', timelimit)
    all_vars = {}
    num_pulls = {}
    T = alg.T
    for i in range(len(alg.arm)):
        num_pulls[i] = []

    for t in range(T):
        for key in num_pulls.keys():
            if alg.action_idxs[t] != key:
                if t != 0:
                    num_pulls[key].append(num_pulls[key][ - 1])
                else:
                    num_pulls[key].append(0)
            else:
                if t != 0:
                    num_pulls[key].append(num_pulls[key][ - 1] + 1)
                else:
                    num_pulls[key].append(1)

    for t in range(T):
        for idx, ele in enumerate(alg.arm):
            if t not in all_vars:
                all_vars[t] = {}
            all_vars[t][idx] = m.addVar(name="u_{}_{}".format(t, idx))

    expr = gp.LinExpr()
    list_of_all_vars = []
    for t in range(T-1,T):
        for i in range(len(alg.arm)):
            list_of_all_vars.append(all_vars[t][i])


    expr.addTerms([1.0] * len(list_of_all_vars), list_of_all_vars)

    # m.setObjective(expr, gp.GRB.MAXIMIZE)



    for t, ele in enumerate(alg.action_idxs):
        for i in range(len(alg.arm)):
            if i != ele and t >= len(alg.arm):
                m.addConstr(all_vars[t-1][ele] + UCB.gcb(T, alg.alpha, num_pulls[ele][t-1]) - all_vars[t-1][i] - UCB.gcb(T, alg.alpha, num_pulls[i][t - 1]) >= 0)
                if t-1 > 0:
                    m.addConstr(all_vars[t][i] - all_vars[t - 1][i] == 0)
            m.addConstr(all_vars[t][i] >= 0)
            m.addConstr(all_vars[t][i] <= 1)

    for t, ele in enumerate(alg.action_idxs):
        if t - 1 >= 0:
            m.addConstr(num_pulls[ele][t] * all_vars[t][ele] - num_pulls[ele][t - 1] * all_vars[t - 1][ele] <= 1)
            m.addConstr(num_pulls[ele][t] * all_vars[t][ele] - num_pulls[ele][t - 1] * all_vars[t - 1][ele] >= 0)
    # m.write("debug.lp")
    m.optimize()
    lp_vals = []

    for i in range(len(alg.arm)):
        lp_vals.append(all_vars[T-1][i].X)
    final_solutions = {t: {i: all_vars[t][i].X for i in range(len(alg.arm))} for t in range(T)}
    # print(lp_vals)

    # if lp_vals.count(0) > 1:
    #     breakpoint()
    return lp_vals


def calc_Vs(alg):
    Vs = []
    Vs.append(alg.lamda * np.identity(alg.dim))
    for t in range(alg.T):
        Vs.append(Vs[-1] + (alg.actions[t].T * alg.actions[t]))
    return(Vs)

def get_orthogonal_matrix(vec):
    first = np.identity(len(vec) - 1)
    if vec[0].item() == 0:
        second = -1 * vec[1:]
    else:
        second = -1 * vec[1:] / vec[0]
    others = np.concatenate((second.T, first),axis=0)
    banana = others @ (others.T @ others) @ others.T
    return banana

def estimate_linucb_means_lp(alg, normalize=True, tolerance=1e-5, timelimit=None):
    m = gp.Model()
    m.Params.LogToConsole = 0
    if timelimit is not None:
        m.setParam('TimeLimit', timelimit)
    all_vars = {}
    T = alg.T

    Vs = calc_Vs(alg)
    for t in range(alg.T):
        all_vars[t] = []
        for i in range(alg.dim):
            all_vars[t].append(m.addVar(name="y_{}_{}".format(t, i)))

    expr = gp.LinExpr()
    expr.addTerms([1.0] * len(all_vars[alg.T - 1]),all_vars[alg.T - 1])

#     m.setObjective(expr, gp.GRB.MAXIMIZE) 
    for t in range(T):
        for action in alg.arm:
            V = Vs[t]
            invV = np.linalg.inv(V)
            if action is not alg.actions[t]:
                expr = gp.LinExpr()

                mult = np.asarray(action @ invV)[0] * -1
                expr.addTerms(mult, all_vars[t])
                constant = np.sqrt(action @ invV @ action.T) * alg.beta * -1
                expr.addConstant(constant)

                opt_mult = np.asarray(alg.actions[t] @ invV)[0]
                expr.addTerms(opt_mult, all_vars[t])
                opt_constant = np.sqrt(alg.actions[t] @ invV @ alg.actions[t].T) * alg.beta
                expr.addConstant(opt_constant)

                m.addConstr(expr >= 0)

                
    for t in range(T):
        if t == T - 1:
            continue
        a_t = alg.actions[t]
        V_t = Vs[t]
        V_tplus1 = Vs[t+1]
        
        V_t_inv = np.linalg.pinv(V_t)
        V_tplus1_inv = np.linalg.pinv(V_tplus1)
        
        orthogonal_projection = get_orthogonal_matrix(V_t_inv @ a_t.T)
        orthogonal_projection = orthogonal_projection/np.max(orthogonal_projection)
        first_coeff = np.asarray(orthogonal_projection @ V_tplus1_inv)
        second_coeff = np.asarray(orthogonal_projection @ V_t_inv)
        
        for curr_dim in range(first_coeff.shape[0]):
            firstexpr = gp.LinExpr()

            if np.any(first_coeff[curr_dim]):
                firstexpr.addTerms(first_coeff[curr_dim], all_vars[t+1])
            if np.any(second_coeff[curr_dim]):
                firstexpr.addTerms(-1 * second_coeff[curr_dim], all_vars[t])
            # breakpoint()
            if np.any(first_coeff[curr_dim]) or np.any(second_coeff[curr_dim]):
                m.addConstr(firstexpr >= -1 * tolerance)
                m.addConstr(firstexpr <= tolerance)
                pass

    for i in range(alg.dim):
        m.addConstr(all_vars[0][i] == 0)



    for i in range(alg.dim):
        for t in range(alg.T):
            if t + 1 < alg.T:
                m.addConstr(all_vars[t][i] + 1 >= all_vars[t + 1][i])
                m.addConstr(all_vars[t][i] <= all_vars[t+1][i])

    m.optimize()

    # breakpoint()
    # active_constraints = []
    # for constr in m.getConstrs():
    #     if abs(constr.slack) < 1e-6:
    #         active_constraints.append(constr)

    # breakpoint()
    final_Vinv = np.linalg.inv(Vs[-1])
    try:
        final_y = np.matrix([ele.X for ele in all_vars[alg.T - 1]]).T
    except:
        return None
    
    theta_estimate = np.linalg.inv(Vs[-1]) @ np.matrix(final_y)
    
    if normalize:
        if np.linalg.norm(theta_estimate) != 0:
            theta_estimate = theta_estimate/np.linalg.norm(theta_estimate)
    
    return theta_estimate

def estimate_phased_elim_subopt_gap(alg, a_star, R, true_theta, l=None):
    if l is None:
        l = alg.max_l - 1
        while alg.eliminated_arms[l].shape[0] < alg.dim:
            l -= 1
    if alg.eliminated_arms[l].shape[0] == 0:
        return -1, None
    # if alg.eliminated_arms[l].shape[0] < alg.dim:
        # raise RuntimeError('Span of arms is insufficient! No estimator can be formed. Span: %d' % alg.eliminated_arms[l].shape[0])

    eliminated_arms = alg.eliminated_arms[l]
    normalized_elim_arms = eliminated_arms / np.linalg.norm(eliminated_arms, axis=1, keepdims=True)
    # print(np.linalg.norm(a_star))
    # print(a_star)
    # print(alg.eliminated_arms[l].shape[0])
    optimal = a_star# / np.linalg.norm(a_star)
    vectors_to_keep = []
    remaining_indices = list(range(normalized_elim_arms.shape[0]))
    # REMOVE THIS
    optimal = alg.mean_active_sets[l][:, np.newaxis] / np.linalg.norm(alg.mean_active_sets[l].shape)
    # print(alg.mean_active_sets[l].shape)
    for i in range(-1, alg.dim-1):
        normalized_elim_arms_2d = normalized_elim_arms[remaining_indices][:, [i, i+1]]
        optimal_2d = optimal[[i, i+1], :]
        cosine_sim = normalized_elim_arms_2d @ optimal_2d
        choice = np.argmin(cosine_sim)
        vectors_to_keep.append(eliminated_arms[remaining_indices][choice])
        # del remaining_indices[choice]
    # print(vectors_to_keep)
    A = np.array(vectors_to_keep)
    # print(A @ true_theta.T)
    # print((R - 6 * 2 **(-l)))
    # print(R)
    #B =
    # vectors_to_keep.append(a_star)
    B = np.ones((alg.dim+1, alg.dim+1), dtype=np.float64)
    B[:-1, 1:] = -1 * A
    B[-1, 1:] = a_star
    print(B)
    print('Inverse estimator has rank %d' % np.linalg.matrix_rank(B))
    delta_a_i = np.ones(alg.dim+1, dtype=np.float64) * 3 * 2**(-l)
    # optimal arm has no suboptimality
    delta_a_i[-1] = 0
    delta_a_i = delta_a_i[:, np.newaxis]

    estimate = (np.linalg.pinv(B) @ (delta_a_i)).T
    mu_estimate = estimate[0, 0]
    theta_estimate = estimate[:, 1:]
    # theta_estimate = (np.linalg.pinv(B) @ 

    # theta_estimate = (np.linalg.pinv(A) @ (np.ones([alg.dim, 1]) * (R - 3*2**(-l)))).T
    theta_estimate = theta_estimate / np.linalg.norm(theta_estimate, ord=1)
    true_theta = true_theta / np.linalg.norm(true_theta, ord=1)
    return np.linalg.norm(theta_estimate - true_theta) / np.linalg.norm(true_theta), theta_estimate, mu_estimate, np.abs(mu_estimate-R)

def estimate_phased_elim_means_slow(alg, a_star, R, true_theta, l=None):
    if l is None:
        l = alg.max_l - 1
        while alg.eliminated_arms[l].shape[0] < alg.dim:
            l -= 1
    if alg.eliminated_arms[l].shape[0] == 0:
        print("gamma insufficient")
        return -1, None
    d = alg.dim
    eliminated_arms = alg.eliminated_arms[l]
    beta = 3*(1-alg.iota)*2**(-l) #depends on omega, but this is an upper bound
    gamma = 2**(-l+1) / np.linalg.norm(true_theta)**2 / d
    elim_norm = np.linalg.norm(eliminated_arms, axis=-1)
    star_norm = np.linalg.norm(a_star)
    a_star = a_star.T
    def proj(a, i):
        e_i = np.zeros((d, 1))
        e_i[i] = 1
        component = ((e_i.T @ a_star)[0, 0] / np.linalg.norm(a_star) * a_star)
        s_i = e_i - component
        proj_matrix = np.zeros((d, 2))
        proj_matrix[:, 0:1] = a_star
        proj_matrix[:, 1:2] = s_i
        projected_coords = a @ proj_matrix #[N x D] x [D x 2] => [N x 2]
        projections = (projected_coords[:, 0] / star_norm / elim_norm)[:, np.newaxis] * a_star.T
        projections += (projected_coords[:, 1] / np.linalg.norm(s_i) / elim_norm)[:, np.newaxis] * s_i.T
        return projections

    vectors_to_keep = []
    for i in range(alg.dim):
        projections = proj(eliminated_arms, i)
        # print(projections.shape)
        # print(a_star.shape)
        inner_prod = projections @ a_star
        # print(inner_prod.shape)
        # workaround dim not squeezing for some reason
        # inner_prod = inner_prod.copy().squeeze(1)
        np.save("temp.npy", inner_prod)
        inner_prod = np.load("temp.npy").squeeze(1)


        # print(np.linalg.norm(projections, axis=-1).shape)
        # print((inner_prod / np.linalg.norm(projections, axis=-1) / np.linalg.norm(a_star)).shape)
        # print((inner_prod / np.linalg.norm(projections, axis=-1)).shape)# / np.linalg.norm(a_star)).shape)
        tau = np.arccos(inner_prod / np.linalg.norm(projections, axis=-1) / np.linalg.norm(a_star))#.squeeze(0)
        # print(tau.shape)
        dist = np.linalg.norm(projections - eliminated_arms, axis=-1)
        # print(dist.shape)
        indexing = (tau > beta) * (dist <= gamma)
        # find valid dist ones?
        # possible = np.where(dist <= gamma)
        # possible_taus = tau[possible]
        # print(len(possible_taus))
        # index = np.argmax(possible_taus)
        # print(index)
        # a_i = possible[index]
        print("%d possible arms for dim %d" % (np.sum(indexing), i))
        # arbitrarily grab the first valid arm
        if np.sum(indexing) == 0:
            print("Failed to find proper arms... possible error")
            if np.sum(dist <= gamma) == 0:
                a_i = np.random.choice(np.where(tau > beta)[0])#[0]
            else:
                a_i = np.random.choice(np.where(dist <= gamma)[0])#[0]
            # print(np.sum(dist <= gamma))
            # print(np.sum(tau > beta))
        else:
            a_i = np.random.choice(np.where(indexing)[0])#[0]
        vectors_to_keep.append(eliminated_arms[a_i])

    print(vectors_to_keep)
    A = np.array(vectors_to_keep)
    print('Inverse estimator has rank %d' % np.linalg.matrix_rank(A))
    theta_estimate = (np.linalg.pinv(A) @ (np.ones([alg.dim, 1]) * (R - 3*2**(-l)))).T
    theta_estimate = theta_estimate / np.linalg.norm(theta_estimate, ord=1)
    print(np.linalg.norm(theta_estimate - true_theta) / np.linalg.norm(true_theta))
    true_theta = true_theta / np.linalg.norm(true_theta, ord=1)
    return np.linalg.norm(theta_estimate - true_theta) / np.linalg.norm(true_theta), theta_estimate



def estimate_phased_elim_means(alg, a_star, R, true_theta, l=None):
    if l is None:
        l = alg.max_l - 1
        while alg.eliminated_arms[l].shape[0] < alg.dim:
            l -= 1
    if alg.eliminated_arms[l].shape[0] == 0:
        print("no eliminated arms?")
        return -1, None
    # if alg.eliminated_arms[l].shape[0] < alg.dim:
        # raise RuntimeError('Span of arms is insufficient! No estimator can be formed. Span: %d' % alg.eliminated_arms[l].shape[0])

    eliminated_arms = alg.eliminated_arms[l]
    normalized_elim_arms = eliminated_arms / np.linalg.norm(eliminated_arms, axis=1, keepdims=True)
    # print(np.linalg.norm(a_star))
    # print(a_star)
    # print(alg.eliminated_arms[l].shape[0])
    optimal = a_star# / np.linalg.norm(a_star)
    vectors_to_keep = []
    remaining_indices = list(range(normalized_elim_arms.shape[0]))
    # REMOVE THIS
    # optimal = alg.mean_active_sets[l][:, np.newaxis] / np.linalg.norm(alg.mean_active_sets[l].shape)
    # print(alg.mean_active_sets[l].shape)
    for i in range(-1, alg.dim-1):
        normalized_elim_arms_2d = normalized_elim_arms[remaining_indices][:, [i, i+1]]
        optimal_2d = optimal[[i, i+1], :]
        cosine_sim = normalized_elim_arms_2d @ optimal_2d
        choice = np.argmin(cosine_sim)
        vectors_to_keep.append(eliminated_arms[remaining_indices][choice])
        # del remaining_indices[choice]
    # print(vectors_to_keep)
    A = np.array(vectors_to_keep)
    print('Inverse estimator has rank %d' % np.linalg.matrix_rank(A))
    # print(A @ true_theta.T)
    # print((R - 6 * 2 **(-l)))
    # print(R)
    theta_estimate = (np.linalg.pinv(A) @ (np.ones([alg.dim, 1]) * (R - 3*2**(-l)))).T
    theta_estimate = theta_estimate / np.linalg.norm(theta_estimate, ord=1)
    true_theta = true_theta / np.linalg.norm(true_theta, ord=1)
    return np.linalg.norm(theta_estimate - true_theta) / np.linalg.norm(true_theta), theta_estimate



def estimate_phased_elim_means_old(alg, a_star, R, true_theta, l=None):
    return None, None
    print('Phase analyzed:')
    print(l)
    eliminated_arms = alg.eliminated_arms[l-1]
    for i, arm in enumerate(eliminated_arms):
        # normalize
        eliminated_arms[i] = arm / np.linalg.norm(eliminated_arms)
    mu = alg.mean_active_sets[l]
    elim_span = []
    theta = np.arccos(R - 3*2**(-l))
    # theta = np.arccos(6*2**(-l))
    # return the diff for bisecting
    def rotation_solver(gamma, i, mu, theta):
        new_vec = copy.deepcopy(mu)
        to_rotate = mu[[i, i+1]]
        rotated = np.array([[np.cos(gamma), np.sin(gamma)], [-np.sin(gamma), np.cos(gamma)]]) @ to_rotate[:, np.newaxis]
        new_vec[[i, i+1]] = rotated.T[0]
        dot_prod = np.clip(mu[np.newaxis, :] @ new_vec / np.linalg.norm(mu) / np.linalg.norm(new_vec), -1, 1)[0]
        # if gamma == np.pi:
            # print(np.arccos(dot_prod) - theta)
        ret = np.arccos(dot_prod) - theta
        return ret
    try:
        for i in range(-1, alg.dim-1):
            gamma = scipy.optimize.bisect(rotation_solver, 0, np.pi, args=(i, mu, theta))
            #print(gamma)
            new_vec = copy.deepcopy(mu)
            to_rotate = mu[[i, i+1]]
            rotated = np.array([[np.cos(gamma), np.sin(gamma)], [-np.sin(gamma), np.cos(gamma)]]) @ to_rotate[:, np.newaxis]
            new_vec[[i, i+1]] = rotated.T[0]
            #print(np.sum(new_vec * mu) / np.linalg.norm(new_vec) / np.linalg.norm(mu))
            # closest_idx = np.argmax((eliminated_arms @ new_vec).flatten())
            #print(new_vec)
            elim_span.append(new_vec)
            #print(closest_idx)
            #elim_span.append(eliminated_arms[closest_idx])
        A = np.array(elim_span)
        theta_estimate = (np.linalg.pinv(A) @ (np.ones([alg.dim, 1]) * (R - 3*2**(-l)))).T
        theta_estimate = theta_estimate / np.linalg.norm(theta_estimate, ord=1)
        true_theta = true_theta / np.linalg.norm(true_theta, ord=1)
        return np.linalg.norm(theta_estimate - true_theta) / np.linalg.norm(true_theta), theta_estimate
    except:
        print('Endpoints were the same signs:')
        print(theta)
        for k in range(33):
            print(rotation_solver(k * np.pi/16, i, mu, theta))
        # print(rotation_solver(np.pi, i, mu, theta))
        return None, None
