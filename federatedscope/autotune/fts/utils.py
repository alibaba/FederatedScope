from datetime import datetime
import GPy
import pickle
import time
import numpy as np
import warnings

from scipydirect import direct

warnings.filterwarnings('ignore')

ERROR_MESSAGES = (
    'u[i] < l[i] for some i',
    'maxf is too large',
    'Initialization failed',
    'There was an error in the creation of the sample points',
    'An error occured while the function was sampled',
    'Maximum number of levels has been reached.',
)
SUCCESS_MESSAGES = (
    'Number of function evaluations done is larger then maxf',
    'Number of iterations is equal to maxT',
    'The best function value found is within fglper of the (known) '
    'global optimum',
    'The volume of the hyperrectangle with best function value found is '
    'below volper',
    'The volume of the hyperrectangle with best function value found is '
    'smaller then volper')


class OptimizeResult(dict):
    """
    Represents the optimization result.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join(
                [k.rjust(m) + ': ' + repr(v) for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"


def minimize(func,
             bounds=None,
             para_dict={},
             nvar=None,
             args=(),
             disp=False,
             eps=1e-4,
             maxf=20000,
             maxT=6000,
             algmethod=0,
             fglobal=-1e100,
             fglper=0.01,
             volper=-1.0,
             sigmaper=-1.0,
             **kwargs):
    if bounds is None:
        lb = np.zeros(nvar, dtype=np.float64)
        ub = np.ones(nvar, dtype=np.float64)
    else:
        bounds = np.asarray(bounds)
        lb = bounds[:, 0]
        ub = bounds[:, 1]

    def _objective_wrap(x, iidata, ddata, cdata, n, iisize, idsize, icsize):
        return func(x, para_dict), 0

    # Dummy values so that the python wrapper will comply with the required
    # signature of the fortran library.
    iidata = np.ones(0, dtype=np.int32)
    ddata = np.ones(0, dtype=np.float64)
    cdata = np.ones([0, 40], dtype=np.uint8)

    x, fun, ierror = direct(_objective_wrap, eps, maxf, maxT, lb, ub,
                            algmethod, 'dummylogfile', fglobal, fglper, volper,
                            sigmaper, iidata, ddata, cdata, disp)

    return OptimizeResult(x=x,
                          fun=fun,
                          status=ierror,
                          success=ierror > 0,
                          message=SUCCESS_MESSAGES[ierror - 1]
                          if ierror > 0 else ERROR_MESSAGES[abs(ierror) - 1])


class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """
    def __init__(self, kind):
        if kind not in ['ts']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose ucb or ts.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    # This function is defined to work with the DIRECT optimizer
    def utility(self, x, para_dict):
        M = para_dict["M"]
        random_features = para_dict["random_features"]
        w_sample = para_dict["w_sample"]
        return self._ts(x, M, random_features, w_sample)

    @staticmethod
    def _ts(x, M, random_features, w_sample):

        s = random_features["s"]
        b = random_features["b"]
        v_kernel = random_features["v_kernel"]

        x = np.squeeze(x).reshape(1, -1)
        features = np.sqrt(2 / M) * np.cos(np.squeeze(np.dot(x, s.T)) + b)
        features = features.reshape(-1, 1)

        features = features / np.sqrt(
            np.inner(np.squeeze(features), np.squeeze(features)))
        features = np.sqrt(v_kernel) * features
        # v_kernel is set to be 1 here in the synthetic experiments

        f_value = np.squeeze(np.dot(w_sample, features))
        optimizer_flag = 1

        return optimizer_flag * f_value


def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking

    :param a: array to trim repeated rows from

    :return: mask of unique rows
    """

    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]


class BColours(object):
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'


class PrintLog(object):
    def __init__(self, params):

        self.ymax = None
        self.xmax = None
        self.params = params
        self.ite = 1

        self.start_time = datetime.now()
        self.last_round = datetime.now()

        # sizes of parameters name and all
        self.sizes = [max(len(ps), 7) for ps in params]

        # Sorted indexes to access parameters
        self.sorti = sorted(range(len(self.params)),
                            key=self.params.__getitem__)

    def reset_timer(self):
        self.start_time = datetime.now()
        self.last_round = datetime.now()

    def print_header(self, initialization=True):

        if initialization:
            print("{}Initialization{}".format(BColours.RED, BColours.ENDC))
        else:
            print("{}Bayesian Optimization{}".format(BColours.RED,
                                                     BColours.ENDC))

        print(BColours.BLUE + "-" * (29 + sum([s + 5 for s in self.sizes])) +
              BColours.ENDC)

        print("{0:>{1}}".format("Step", 5), end=" | ")
        print("{0:>{1}}".format("Time", 6), end=" | ")
        print("{0:>{1}}".format("Value", 10), end=" | ")

        for index in self.sorti:
            print("{0:>{1}}".format(self.params[index], self.sizes[index] + 2),
                  end=" | ")
        print('')

    def print_step(self, x, y, warning=False):

        print("{:>5d}".format(self.ite), end=" | ")

        m, s = divmod((datetime.now() - self.last_round).total_seconds(), 60)
        print("{:>02d}m{:>02d}s".format(int(m), int(s)), end=" | ")

        if self.ymax is None or self.ymax < y:
            self.ymax = y
            self.xmax = x
            print("{0}{2: >10.5f}{1}".format(BColours.MAGENTA, BColours.ENDC,
                                             y),
                  end=" | ")

            for index in self.sorti:
                print("{0}{2: >{3}.{4}f}{1}".format(
                    BColours.GREEN, BColours.ENDC, x[index],
                    self.sizes[index] + 2, min(self.sizes[index] - 3, 6 - 2)),
                      end=" | ")
        else:
            print("{: >10.5f}".format(y), end=" | ")
            for index in self.sorti:
                print("{0: >{1}.{2}f}".format(x[index], self.sizes[index] + 2,
                                              min(self.sizes[index] - 3, 4)),
                      end=" | ")

        if warning:
            print("{}Warning: Test point chose at "
                  "random due to repeated sample.{}".format(
                      BColours.RED, BColours.ENDC))

        print()

        self.last_round = datetime.now()
        self.ite += 1

    def print_summary(self):
        pass


def acq_max(
    ac,
    gp,
    M,
    N,
    random_features,
    info_ts,
    pt,
    ws,
    use_target_label,
    w_sample,
    y_max,
    bounds,
    iteration,
    gp_samples=None,
):
    para_dict = {
        "gp": gp,
        "M": M,
        "N": N,
        "random_features": random_features,
        "info_ts": info_ts,
        "pt": pt,
        "ws": ws,
        "use_target_label": use_target_label,
        "tmp_ucb": None,
        "w_sample": w_sample,
        "y_max": y_max,
        "iteration": iteration,
        "gp_samples": gp_samples,
    }

    x_tries = np.random.uniform(bounds[:, 0],
                                bounds[:, 1],
                                size=(10000, bounds.shape[0]))
    ys = []
    for x in x_tries:
        ys.append(ac(x.reshape(1, -1), para_dict))
    ys = np.array(ys)

    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()
    _num_trials = 1
    x_seeds = np.random.uniform(bounds[:, 0],
                                bounds[:, 1],
                                size=(_num_trials, bounds.shape[0]))
    for x_try in x_seeds:
        res = minimize(lambda _x, para_dict: -ac(_x.reshape(1, -1), para_dict),
                       para_dict=para_dict,
                       bounds=bounds,
                       method="L-BFGS-B")
        if max_acq is None or -res.fun >= max_acq:
            x_max = res.x
            max_acq = -res.fun

    return x_max, None


def x2conf(x, pbounds, ss):
    x = np.array(x).reshape(-1)
    assert len(x) == len(pbounds)
    params = {}

    for i, (k, b) in zip(range(len(x)), pbounds.items()):
        p_inst = ss[k]
        l, u = b
        p = float(1. * x[i] * (u - l) + l)
        if p_inst.log:
            p = 10**p
        params[k] = int(p) if 'int' in str(type(p_inst)).lower() else p
    return params


class LocalBO(object):
    def __init__(self,
                 cid,
                 f,
                 bounds,
                 keys,
                 gp_opt_schedule=5,
                 ARD=False,
                 use_init=False,
                 log_file=None,
                 save_init=False,
                 save_init_file=None,
                 N=None,
                 info_ts=None,
                 pt=None,
                 ls=None,
                 var=None,
                 g_var=None,
                 P_N=None,
                 M_target=100,
                 verbose=1):
        """
        f: the objective function of the target agent
        pbounds: dict of hyperparameter ranges e.g. {'lr':(lower, upper)}
        gp_opt_schedule: optimize the GP hyperparameters after every
        gp_opt_schedule iterations
        M: the number of random features for RFF approximation
        N: the number of other agents
        random_features: the saved random features, which is shared among
        all agents info_ts: the information received from each agent
        for the algorithm; for each agent,
        the information includes a sampled \omega_n
        pt: the sequence p_t; to run the standard TS algorithm
        (without using information from other agents), just set pt to all 1
        P_N: the distribution over agents used by the FTS algorithm
        M_target: the number of random features used by both TS and FTS
        to draw samples from the GP posterior of the target agent
        """

        self.N = N
        self.info_ts = info_ts
        self.M_target = M_target
        self.agents_used_flag = np.ones(self.N)
        self.pt = pt
        self.ws = P_N
        self.use_init = use_init
        self.time_started = 0
        self.ARD = ARD
        self.log_file = log_file
        self.incumbent = None
        self.keys = keys
        self.dim = len(keys)
        self.bounds = np.asarray(bounds)
        self.f = f
        self.initialized = False
        self.init_points = []
        self.x_init = []
        self.y_init = []
        self.X = np.array([]).reshape(-1, 1)
        self.Y = np.array([])
        self.i = 0
        self.gp = None
        self.gp_params = None
        self.gp_opt_schedule = gp_opt_schedule
        self.federated_gps = []
        self.util = None
        self.plog = PrintLog(self.keys)
        self.save_init = save_init
        self.save_init_file = save_init_file
        self.ls = ls
        self.var = var
        self.g_var = g_var
        self.res = {}
        self.res['max'] = {'max_val': None, 'max_params': None}
        self.res['all'] = {
            'values': [],
            'params': [],
            'init_values': [],
            'init_params': [],
            'init': [],
            'time_started': 0,
            'timestamps': []
        }
        self.cid = cid
        self.verbose = verbose

    def sample_from_local(self, y_max, iteration):
        M_target = self.M_target

        ls_target = self.gp["rbf.lengthscale"][0]
        v_kernel = self.gp["rbf.variance"][0]
        obs_noise = self.gp["Gaussian_noise.variance"][0]

        s = np.random.multivariate_normal(
            np.zeros(self.dim), 1 / (ls_target**2) * np.identity(self.dim),
            M_target)
        b = np.random.uniform(0, 2 * np.pi, M_target)

        random_features_target = {
            "M": M_target,
            "length_scale": ls_target,
            "s": s,
            "b": b,
            "obs_noise": obs_noise,
            "v_kernel": v_kernel
        }

        Phi = np.zeros((self.X.shape[0], M_target))
        for i, x in enumerate(self.X):
            x = np.squeeze(x).reshape(1, -1)
            features = np.sqrt(
                2 / M_target) * np.cos(np.squeeze(np.dot(x, s.T)) + b)
            features = features / np.sqrt(np.inner(features, features))
            features = np.sqrt(v_kernel) * features
            Phi[i, :] = features

        Sigma_t = np.dot(Phi.T, Phi) + obs_noise * np.identity(M_target)
        Sigma_t_inv = np.linalg.inv(Sigma_t)
        nu_t = np.dot(np.dot(Sigma_t_inv, Phi.T), self.Y.reshape(-1, 1))
        w_sample = np.random.multivariate_normal(np.squeeze(nu_t),
                                                 obs_noise * Sigma_t_inv, 1)

        x_max, all_ucb = acq_max(ac=self.util_ts.utility,
                                 gp=self.gp,
                                 M=M_target,
                                 N=self.N,
                                 gp_samples=None,
                                 random_features=random_features_target,
                                 info_ts=self.info_ts,
                                 pt=self.pt,
                                 ws=self.ws,
                                 use_target_label=True,
                                 w_sample=w_sample,
                                 y_max=y_max,
                                 bounds=self.bounds,
                                 iteration=iteration)
        return x_max, all_ucb

    def maximize(self, init_points=5, n_iter=25):
        assert init_points > 1
        self.plog.reset_timer()
        self.res['all']['time_started'] = time.time()
        self.util_ts = UtilityFunction(kind="ts")
        # get random initializations
        if not self.initialized:
            if self.use_init is not None:
                init = pickle.load(open(self.use_init, "rb"))
                self.X, self.Y = init["X"], init["Y"]
                self.incumbent = np.max(self.Y)
                self.initialized = True
                self.res['all']['init'] = init
                self.res['all']['init_values'] = list(self.Y)
                print("==> Using pre-existing initializations "
                      "with {0} points".format(len(self.Y)))
            else:
                if init_points > 0:
                    print('==> Random initialize')
                    ls = [
                        np.random.uniform(x[0], x[1], size=init_points)
                        for x in self.bounds
                    ]

                    self.init_points += list(map(list, zip(*ls)))
                    y_init = []
                    for x in self.init_points:
                        print("[init point]: ", x)
                        curr_y = self.f(x)
                        y_init.append(curr_y)
                        self.res['all']['init_values'].append(curr_y)
                        self.res['all']['init_params'].append(
                            dict(zip(self.keys, x)))

                    self.X = np.asarray(self.init_points)
                    self.Y = np.asarray(y_init)
                    self.incumbent = np.max(y_init)
                    self.initialized = True
                    init = {"X": self.X, "Y": self.Y}
                    self.res['all']['init'] = init

                    if self.save_init:
                        pickle.dump(init, open(self.save_init_file, "wb"))

        y_max = np.max(self.Y)
        ur = unique_rows(self.X)

        self.gp = GPy.models.GPRegression(
            self.X[ur], self.Y[ur].reshape(-1, 1),
            GPy.kern.RBF(input_dim=self.X.shape[1],
                         lengthscale=self.ls,
                         variance=self.var,
                         ARD=self.ARD))
        self.gp["Gaussian_noise.variance"][0] = self.g_var
        print("---Client %d initial hyper: " % self.cid, self.gp)

        x_max, all_ucb = self.sample_from_local(y_max, 1)

        if self.verbose:
            self.plog.print_header(initialization=False)

        for i in range(n_iter):
            if not self.X.shape[0] == 0:
                if np.any(np.all(self.X - x_max == 0, axis=1)):
                    x_max = np.random.uniform(self.bounds[:, 0],
                                              self.bounds[:, 1],
                                              size=self.bounds.shape[0])

            curr_y = self.f(x_max)
            self.res["all"]["timestamps"].append(time.time())
            self.Y = np.append(self.Y, curr_y)
            self.X = np.vstack((self.X, x_max.reshape((1, -1))))
            if self.Y[-1] > y_max:
                y_max = self.Y[-1]
                self.incumbent = self.Y[-1]
            ur = unique_rows(self.X)
            self.gp.set_XY(X=self.X[ur], Y=self.Y[ur].reshape(-1, 1))

            if i >= self.gp_opt_schedule and i % self.gp_opt_schedule == 0:
                self.gp.optimize_restarts(num_restarts=10,
                                          messages=False,
                                          verbose=False)
                self.gp_params = self.gp.parameters
            if i == n_iter - 1:
                print("---Client %d optimized hyper: " % self.cid, self.gp)

            _loop = True
            while _loop:
                try:
                    x_max, all_ucb = self.sample_from_local(y_max, i + 2)
                    _loop = False
                except:
                    _loop = True

            if self.verbose:
                self.plog.print_step(x_max, self.Y[-1], warning=False)

            self.i += 1
            x_max_param = self.X[self.Y.argmax(), :-1]
            self.res['max'] = {
                'max_val': self.Y.max(),
                'max_params': dict(zip(self.keys, x_max_param))
            }
            self.res['all']['values'].append(self.Y[-1])
            self.res['all']['params'].append(self.X[-1])

            if self.log_file is not None:
                pickle.dump(self.res, open(self.log_file, "wb"))

        if self.verbose:
            self.plog.print_summary()
