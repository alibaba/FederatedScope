import numpy as np


def FL(x, objs, sizes):
    cur_x = x
    if not isinstance(sizes, list):
        sizes = len(objs) * [sizes]
    for r in range(5):
        updates = []
        for i, f in enumerate(objs):
            val, grad = f(cur_x)
            updates.append(-1.0 * sizes[i] * grad)
        cur_x += np.mean(updates)
    vals = []
    for i, f in enumerate(objs):
        val, grad = f(cur_x)
        vals.append(val)
    return np.mean(vals)


if __name__=="__main__":
    Fis = []
    for a in [0.02, 0.1, 0.5, 2.5, 12.5]:
        Fis.append(lambda x: (a * x**2, 2*a*x))
    # without personalization
    best = float("inf")
    best_lr = None
    for lr in [0.001, 0.005, 0.025, 0.125, 0.625]:
        results = []
        for i in range(32):
            np.random.seed(i)
            init_x = np.random.uniform(-10.0, 10.0)
            results.append(FL(init_x, Fis, lr))
        print(np.mean(results), lr)
        if best > np.mean(results):
            best = np.mean(results)
            best_lr = lr
    print(best, best_lr)

    # with personalization
    best = float("inf")
    best_lrs = None
    for trial in range(64):
        np.random.seed(trial)
        lrs = np.random.choice([0.001, 0.005, 0.025, 0.125, 0.625], 5)
        results = []
        for _ in range(32):
            np.random.seed(i)
            init_x = np.random.uniform(-10.0, 10.0)
            results.append(FL(init_x, Fis, lrs))
        if best > np.mean(results):
            best = np.mean(results)
            best_lrs = lrs
    print(best, best_lrs)
