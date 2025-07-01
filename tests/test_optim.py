import numpy as np

from rsnn.optim import *


def test_solver_initialization():
    # Initialize the solver
    solver = Solver(np.intp(2))
    assert solver.n_vars == 2
    assert solver.n_cstrs == 0

    # Add a few linear constraints a x <= b
    solver.add_constraint(np.array([1.0, 2.0]), np.array(5.0))
    solver.add_constraint(np.array([3.0, 4.0]), np.array(11.0))
    solver.add_constraint(np.array([2.0, 8.0]), np.array(12.0))
    assert solver.n_vars == 2
    assert solver.n_cstrs == 3

    assert np.allclose(
        np.diag(solver.A @ solver.A.T), np.ones(3)
    )  # Check normalization
    assert np.allclose(solver.Wbxt_N, solver.A @ solver.A.T)  # Check Wbxt_N
    assert np.allclose(solver.xibxt_N, -solver.b)  # Check xibxt_N


def test_solver_dbffd():
    # Initialize the solver
    solver = Solver(10)
    assert solver.n_vars == 10
    assert solver.n_cstrs == 0

    status = solver.dbffd()
    assert status == 1

    # Add a few linear constraints a x <= b
    solver.add_constraint(
        np.array(
            [
                -0.35793998,
                -0.27761938,
                -0.0408258,
                -0.12746512,
                -0.3404259,
                -0.81376922,
                -0.30628137,
                -0.48007243,
                0.02302248,
                0.24069189,
            ]
        ),
        np.array(0.50860874),
    )
    solver.add_constraint(
        np.array(
            [
                0.57031823,
                0.41957217,
                0.62768923,
                -0.48580253,
                -0.14281048,
                -0.12147438,
                0.65654433,
                0.45457545,
                0.23564731,
                -0.1224626,
            ]
        ),
        np.array(1.003075),
    )
    solver.add_constraint(
        np.array(
            [
                0.3951132,
                -0.15796504,
                0.81129171,
                -0.75362291,
                0.7669671,
                0.76668151,
                -0.33074134,
                0.29926723,
                -0.73339186,
                0.8264834,
            ]
        ),
        np.array(-1.62390705),
    )
    solver.add_constraint(
        np.array(
            [
                -0.93735882,
                0.19003541,
                -0.3478004,
                -0.06509792,
                -0.3212192,
                -0.01530734,
                -0.91639121,
                0.5778128,
                -0.18541728,
                -0.46247663,
            ],
        ),
        np.array(-1.5753577),
    )
    assert solver.n_vars == 10
    assert solver.n_cstrs == 4

    status = solver.dbffd()
    assert status is 1
    assert np.isclose(
        solver.cost, 1.4293642842883392
    )  # Check if the cost is below the tolerance
    assert np.allclose(
        solver.x,
        np.array(
            [
                0.49443406,
                -0.05898912,
                -0.16868494,
                0.44813251,
                -0.16515933,
                -0.39468335,
                0.86323825,
                -0.59236274,
                0.52774889,
                -0.09063558,
            ]
        ),
    )
    assert np.all(solver.A @ solver.x - solver.b <= 1e-6)

    # Add a new constraint to the solver, which makes the problem infeasible
    solver.add_constraint(-solver.A[0], -solver.b[0] - 0.1)
    assert solver.n_vars == 10
    assert solver.n_cstrs == 5

    # Perform dual coordinate descent
    status = solver.dbffd()
    assert status == -1  # Status should be -1 for infeasibility
    assert np.isposinf(
        solver.cost
    )  # Cost should be positive infinity for infeasibility

def test_neuron_init_solver():
    np.random.seed(42)  # For reproducibility
    
    neuron = Neuron()

    f_times = np.sort(np.random.uniform(0.0, 80.0, size=20))
    f_times += np.arange(20)

    in_times = np.random.uniform(0.0, 100.0, size=2000)
    in_channels = np.random.randint(0, 100, size=2000)

    neuron.init_solver(
        np.copy(f_times),
        np.copy(in_times),
        np.copy(in_channels),
        period=100.0,
    )

    assert neuron.solver is not None
    assert neuron.solver.n_vars == 100
    assert neuron.solver.n_cstrs == 20


def test_neuron_learn():
    import time

    np.random.seed(42)  # For reproducibility
    
    neuron = Neuron()

    # firing times are separated by at least 1 time unit
    f_times = np.sort(np.random.uniform(0.0, 90.0, size=10))
    f_times += np.arange(10)

    in_times = np.random.uniform(0.0, 100.0, size=2000)
    in_channels = np.random.randint(0, 100, size=2000)

    neuron.init_solver(
        np.copy(f_times),
        np.copy(in_times),
        np.copy(in_channels),
        period=100.0,
    )
    res = neuron.learn(order=1, conv_tol=1e-9)
    assert res == 1
    assert np.isclose(neuron.solver.cost, 3.2043950176058726)

    neuron.init_solver(
        np.copy(f_times),
        np.copy(in_times),
        np.copy(in_channels),
        period=100.0,
    )
    res = neuron.learn(order=2, conv_tol=1e-9)
    assert res == 1
    assert np.isclose(neuron.solver.cost, 3.2043950176058726)
