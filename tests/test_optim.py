import numpy as np

from rsnn.optim import *


def test_find_maximum_violation():
    # Test a few single-interval cases
    res = find_maximum_violation(
        np.array([0.0]), np.array([1.0]), np.array([10.0]), np.array([0.0])
    )
    assert res is not None
    vmax, _, dtmax = res
    assert np.isclose(vmax, 0.36787944117144233)
    assert np.isclose(dtmax, 1.0)

    res = find_maximum_violation(
        np.array([-1.0]), np.array([0.0]), np.array([10.0]), np.array([0.0])
    )
    assert res is None

    res = find_maximum_violation(
        np.array([-1.0]), np.array([1.0]), np.array([10.0]), np.array([0.0])
    )
    assert res is not None
    vmax, _, dtmax = res
    assert np.isclose(vmax, 0.1353352832366127)
    assert np.isclose(dtmax, 2.0)

    res = find_maximum_violation(
        np.array([-1.0]), np.array([-1.0]), np.array([1.0]), np.array([0.0])
    )
    assert res is None

    res = find_maximum_violation(
        np.array([-0.5]), np.array([-1.0]), np.array([1.0]), np.array([0.0])
    )
    assert res is None

    res = find_maximum_violation(
        np.array([-0.5]), np.array([-1.0]), np.array([2.0]), np.array([0.0])
    )
    assert res is None

    res = find_maximum_violation(
        np.array([1.0]), np.array([0.0]), np.array([2.0]), np.array([0.0])
    )
    assert res is not None
    vmax, _, dtmax = res
    assert np.isclose(vmax, 1.0)
    assert np.isclose(dtmax, 0.0)

    res = find_maximum_violation(
        np.array([0.5]), np.array([1.0]), np.array([2.0]), np.array([0.0])
    )
    assert res is not None
    vmax, _, dtmax = res
    assert np.isclose(vmax, 0.6065306597126334)
    assert np.isclose(dtmax, 0.5)

    res = find_maximum_violation(
        np.array([0.0]), np.array([1.0]), np.array([0.0]), np.array([0.0])
    )
    assert res is None

    res = find_maximum_violation(
        np.array([0.0]), np.array([1.0]), np.array([np.inf]), np.array([0.0])
    )
    assert res is not None
    vmax, _, dtmax = res
    assert np.isclose(vmax, 0.36787944117144233)
    assert np.isclose(dtmax, 1.0)

    res = find_maximum_violation(
        np.array([-1.0]), np.array([0.0]), np.array([np.inf]), np.array([0.0])
    )
    assert res is None

    res = find_maximum_violation(
        np.array([0.0]), np.array([1.0]), np.array([2.0]), np.array([-np.inf])
    )
    assert res is not None
    vmax, _, dtmax = res
    assert np.isclose(vmax, np.inf)
    assert np.isclose(dtmax, 0.0)

    res = find_maximum_violation(
        np.array([0.0]), np.array([1.0]), np.array([2.0]), np.array([np.inf])
    )
    assert res is None

    # Test with multiple random intervals, all with the same length
    np.random.seed(42)  # For reproducibility
    res = find_maximum_violation(
        np.random.randn(10), np.random.randn(10), np.ones(10), np.zeros(10)
    )
    assert res is not None
    vmax, _, dtmax = res
    assert np.isclose(vmax, 1.5792128155073915)
    assert np.isclose(dtmax, 0.0)

    res = find_maximum_violation(
        np.random.randn(10), np.random.randn(10), np.full(10, np.inf), np.zeros(10)
    )
    assert res is not None
    vmax, _, dtmax = res
    assert np.isclose(vmax, 1.465648768921554)
    assert np.isclose(dtmax, 0.0)


def test_solver_initialization():
    # Initialize the solver
    solver = QProgram(42)
    assert solver.n_vars == 42
    assert solver.n_cstrs == 0
    assert solver.dual_cost == 0.0
    assert solver.primal_cost == 0.0
    assert solver.xt.shape == (0,)
    assert solver.x.shape == (42,)

    solver.add_constraints(np.random.randn(4, 42), np.random.randn(4))
    assert solver.n_vars == 42
    assert solver.n_cstrs == 4
    assert solver.A.shape == (4, 42)
    assert solver.b.shape == (4,)
    assert np.isnan(solver.dual_cost)
    assert np.isnan(solver.primal_cost)
    assert solver.xt.shape == (4,)
    assert solver.x.shape == (42,)

    solver.add_constraints(np.random.randn(3, 42), np.random.randn(3))
    assert solver.n_vars == 42
    assert solver.n_cstrs == 7

    assert solver.A.shape == (7, 42)
    assert solver.b.shape == (7,)
    assert np.isnan(solver.dual_cost)
    assert np.isnan(solver.primal_cost)
    assert solver.xt.shape == (7,)
    assert solver.x.shape == (42,)

    # assert np.allclose(
    #     np.diag(solver.A @ solver.A.T), np.ones(3)
    # )  # Check normalization
    # assert np.allclose(solver.Wbxt_N, solver.A @ solver.A.T)  # Check Wbxt_N
    # assert np.allclose(solver.xibxt_N, -solver.b)  # Check xibxt_N


def test_solver_dbffd():
    np.random.seed(42)  # For reproducibility

    # Initialize the solver
    solver = QProgram(10)
    assert solver.n_vars == 10
    assert solver.n_cstrs == 0

    # Solve unconstrained problem
    status = solver.dbffd()
    assert status == 1 # optimal
    assert np.isclose(solver.dual_cost, 0.0)
    assert np.isclose(solver.primal_cost, 0.0)
    assert np.allclose(solver.xt, np.zeros(0))
    assert np.allclose(solver.x, np.zeros(10))

    # Create a random feasible problem
    A = np.random.randn(4, 10)
    b = A @ np.random.randn(10) + np.random.rand(4)
    solver.add_constraints(A, b)
    assert solver.n_cstrs == 4
    status = solver.dbffd()
    assert status == 1  # optimal
    assert np.isclose(solver.dual_cost, 0.25631439311777243)
    assert np.isclose(solver.primal_cost, 0.25631439311777243)

    # Add a few more feasible random constraints to the solver
    A = np.random.randn(2, 10)
    b = A @ np.random.randn(10) + np.random.rand(2)
    solver.add_constraints(A, b)
    assert solver.n_cstrs == 6
    status = solver.dbffd()
    assert status == 1  # optimal
    assert np.isclose(solver.dual_cost, 0.89746396022)
    assert np.isclose(solver.primal_cost, 0.89746396022)

    # Add a trivial constraint that does not change the solution
    A = np.zeros((1, 10))
    b = np.random.rand(1)
    solver.add_constraints(A, b)
    assert solver.n_cstrs == 7
    status = solver.dbffd()
    assert status == 1  # optimal
    assert np.isclose(solver.dual_cost, 0.89746396022)
    assert np.isclose(solver.primal_cost, 0.89746396022)

    # Add a few more feasible random constraints to the solver
    A = np.random.randn(1, 10)
    b = A @ np.random.randn(10) + np.random.rand(1)
    solver.add_constraints(A, b)
    assert solver.n_cstrs == 8
    status = solver.dbffd()
    assert status == 1  # optimal
    assert np.isclose(solver.dual_cost, 0.89746396022)
    assert np.isclose(solver.primal_cost, 0.89746396022)

    # Add an additional constraint that makes the problem infeasible
    solver.add_constraints(-A, -b - np.random.rand(1))
    assert solver.n_cstrs == 9
    status = solver.dbffd()
    assert status == -1  # infeasible
    assert np.isposinf(solver.dual_cost)
    assert np.isnan(solver.primal_cost)
    assert np.all(np.isnan(solver.xt))
    assert np.all(np.isnan(solver.x))


# def test_solver_dbffd_beta_inf():
#     np.random.seed(42)  # For reproducibility

#     # Initialize the solver
#     solver = QProgram(10)
#     assert solver.n_vars == 10
#     assert solver.n_cstrs == 0

#     # Solve unconstrained problem
#     solver.beta.fill(np.inf)
#     status = solver.dbffd()
#     assert status == 1 # optimal
#     assert np.isclose(solver.dual_cost, 0.0)
#     assert np.isclose(solver.primal_cost, 0.0)
#     assert np.allclose(solver.xt, np.zeros(0))
#     assert np.allclose(solver.x, np.zeros(10))

#     # Create a random feasible problem
#     A = np.random.randn(4, 10)
#     b = A @ np.random.randn(10) + np.random.rand(4)
#     solver.add_constraints(A, b)
#     assert solver.n_cstrs == 4
#     solver.beta.fill(np.inf)
#     status = solver.dbffd()
#     assert status == 1  # optimal
#     assert np.isclose(solver.dual_cost, 0.25631439311777243)
#     assert np.isclose(solver.primal_cost, 0.25631439311777243)

#     # Add a few more feasible random constraints to the solver
#     A = np.random.randn(2, 10)
#     b = A @ np.random.randn(10) + np.random.rand(2)
#     solver.add_constraints(A, b)
#     assert solver.n_cstrs == 6
#     solver.beta.fill(np.inf)
#     status = solver.dbffd()
#     assert status == 1  # optimal
#     assert np.isclose(solver.dual_cost, 0.89746396022)
#     assert np.isclose(solver.primal_cost, 0.89746396022)

#     # Add a trivial constraint that does not change the solution
#     A = np.zeros((1, 10))
#     b = np.random.rand(1)
#     solver.add_constraints(A, b)
#     assert solver.n_cstrs == 7
#     solver.beta.fill(np.inf)
#     status = solver.dbffd()
#     assert status == 1  # optimal
#     assert np.isclose(solver.dual_cost, 0.89746396022)
#     assert np.isclose(solver.primal_cost, 0.89746396022)

#     # Add a few more feasible random constraints to the solver
#     A = np.random.randn(1, 10)
#     b = A @ np.random.randn(10) + np.random.rand(1)
#     solver.add_constraints(A, b)
#     assert solver.n_cstrs == 8
#     solver.beta.fill(np.inf)
#     status = solver.dbffd()
#     assert status == 1  # optimal
#     assert np.isclose(solver.dual_cost, 0.89746396022)
#     assert np.isclose(solver.primal_cost, 0.89746396022)

#     # Add an additional constraint that makes the problem infeasible
#     solver.add_constraints(-A, -b - np.random.rand(1))
#     assert solver.n_cstrs == 9
#     solver.beta.fill(np.inf)
#     status = solver.dbffd()
#     assert status == -1  # infeasible
#     assert np.isposinf(solver.dual_cost)
#     assert np.isnan(solver.primal_cost)
#     assert np.all(np.isnan(solver.xt))
#     assert np.all(np.isnan(solver.x))

# def test_solver_dpcd():
#     np.random.seed(42)  # For reproducibility

#     # Initialize the solver
#     solver = QProgram(10)
#     assert solver.n_vars == 10
#     assert solver.n_cstrs == 0

#     # Solve unconstrained problem
#     status = solver.dpcd()
#     assert status == 1 # optimal
#     assert np.isclose(solver.dual_cost, 0.0)
#     assert np.isclose(solver.primal_cost, 0.0)
#     assert np.allclose(solver.xt, np.zeros(0))
#     assert np.allclose(solver.x, np.zeros(10))

#     # Create a random feasible problem
#     A = np.random.randn(4, 10)
#     b = A @ np.random.randn(10) + np.random.rand(4)
#     solver.add_constraints(A, b)
#     assert solver.n_cstrs == 4
#     status = solver.dpcd()
#     assert status == 1  # optimal
#     assert np.isclose(solver.dual_cost, 0.25631439311777243)
#     assert np.isclose(solver.primal_cost, 0.25631439311777243)

#     # Add a few more feasible random constraints to the solver
#     A = np.random.randn(2, 10)
#     b = A @ np.random.randn(10) + np.random.rand(2)
#     solver.add_constraints(A, b)
#     assert solver.n_cstrs == 6
#     status = solver.dpcd()
#     assert status == 1  # optimal
#     assert np.isclose(solver.dual_cost, 0.89746396022)
#     assert np.isclose(solver.primal_cost, 0.89746396022)

#     # Add a trivial constraint that does not change the solution
#     A = np.zeros((1, 10))
#     b = np.random.rand(1)
#     solver.add_constraints(A, b)
#     assert solver.n_cstrs == 7
#     status = solver.dpcd()
#     assert status == 1  # optimal
#     assert np.isclose(solver.dual_cost, 0.89746396022)
#     assert np.isclose(solver.primal_cost, 0.89746396022)

#     # Add a few more feasible random constraints to the solver
#     A = np.random.randn(1, 10)
#     b = A @ np.random.randn(10) + np.random.rand(1)
#     solver.add_constraints(A, b)
#     assert solver.n_cstrs == 8
#     status = solver.dpcd()
#     assert status == 1  # optimal
#     assert np.isclose(solver.dual_cost, 0.89746396022)
#     assert np.isclose(solver.primal_cost, 0.89746396022)

#     # Add an additional constraint that makes the problem infeasible
#     solver.add_constraints(-A, -b - np.random.rand(1))
#     assert solver.n_cstrs == 9
#     status = solver.dpcd()
#     assert status == -1  # infeasible
#     assert np.isposinf(solver.dual_cost)
#     assert np.isnan(solver.primal_cost)
#     assert np.all(np.isnan(solver.xt))
#     assert np.all(np.isnan(solver.x))

    # # Add a few linear constraints a x <= b
    # solver.add_constraint(
    #     np.array(
    #         [
    #             -0.35793998,
    #             -0.27761938,
    #             -0.0408258,
    #             -0.12746512,
    #             -0.3404259,
    #             -0.81376922,
    #             -0.30628137,
    #             -0.48007243,
    #             0.02302248,
    #             0.24069189,
    #         ]
    #     ),
    #     np.array(0.50860874),
    # )
    # solver.add_constraint(
    #     np.array(
    #         [
    #             0.57031823,
    #             0.41957217,
    #             0.62768923,
    #             -0.48580253,
    #             -0.14281048,
    #             -0.12147438,
    #             0.65654433,
    #             0.45457545,
    #             0.23564731,
    #             -0.1224626,
    #         ]
    #     ),
    #     np.array(1.003075),
    # )
    # solver.add_constraint(
    #     np.array(
    #         [
    #             0.3951132,
    #             -0.15796504,
    #             0.81129171,
    #             -0.75362291,
    #             0.7669671,
    #             0.76668151,
    #             -0.33074134,
    #             0.29926723,
    #             -0.73339186,
    #             0.8264834,
    #         ]
    #     ),
    #     np.array(-1.62390705),
    # )
    # solver.add_constraint(
    #     np.array(
    #         [
    #             -0.93735882,
    #             0.19003541,
    #             -0.3478004,
    #             -0.06509792,
    #             -0.3212192,
    #             -0.01530734,
    #             -0.91639121,
    #             0.5778128,
    #             -0.18541728,
    #             -0.46247663,
    #         ],
    #     ),
    #     np.array(-1.5753577),
    # )
    # assert solver.n_vars == 10
    # assert solver.n_cstrs == 4

    # status = solver.dbffd()
    # assert status is 1
    # assert np.isclose(
    #     solver.cost, 1.4293642842883392
    # )  # Check if the cost is below the tolerance
    # assert np.allclose(
    #     solver.x,
    #     np.array(
    #         [
    #             0.49443406,
    #             -0.05898912,
    #             -0.16868494,
    #             0.44813251,
    #             -0.16515933,
    #             -0.39468335,
    #             0.86323825,
    #             -0.59236274,
    #             0.52774889,
    #             -0.09063558,
    #         ]
    #     ),
    # )
    # assert np.all(solver.A @ solver.x - solver.b <= 1e-6)

    # # Add a new constraint to the solver, which makes the problem infeasible
    # solver.add_constraint(-solver.A[0], -solver.b[0] - 0.1)
    # assert solver.n_vars == 10
    # assert solver.n_cstrs == 5

    # # Perform dual coordinate descent
    # status = solver.dbffd()
    # assert status == -1  # Status should be -1 for infeasibility
    # assert np.isposinf(
    #     solver.cost
    # )  # Cost should be positive infinity for infeasibility


# def test_solver_dpcd():
#     # Initialize the solver
#     solver = QProgram(10)
#     assert solver.n_vars == 10
#     assert solver.n_cstrs == 0

#     status = solver.dpcd()
#     assert status == 1

#     # Add a few linear constraints a x <= b
#     solver.add_constraint(
#         np.array(
#             [
#                 -0.35793998,
#                 -0.27761938,
#                 -0.0408258,
#                 -0.12746512,
#                 -0.3404259,
#                 -0.81376922,
#                 -0.30628137,
#                 -0.48007243,
#                 0.02302248,
#                 0.24069189,
#             ]
#         ),
#         np.array(0.50860874),
#     )
#     solver.add_constraint(
#         np.array(
#             [
#                 0.57031823,
#                 0.41957217,
#                 0.62768923,
#                 -0.48580253,
#                 -0.14281048,
#                 -0.12147438,
#                 0.65654433,
#                 0.45457545,
#                 0.23564731,
#                 -0.1224626,
#             ]
#         ),
#         np.array(1.003075),
#     )
#     solver.add_constraint(
#         np.array(
#             [
#                 0.3951132,
#                 -0.15796504,
#                 0.81129171,
#                 -0.75362291,
#                 0.7669671,
#                 0.76668151,
#                 -0.33074134,
#                 0.29926723,
#                 -0.73339186,
#                 0.8264834,
#             ]
#         ),
#         np.array(-1.62390705),
#     )
#     solver.add_constraint(
#         np.array(
#             [
#                 -0.93735882,
#                 0.19003541,
#                 -0.3478004,
#                 -0.06509792,
#                 -0.3212192,
#                 -0.01530734,
#                 -0.91639121,
#                 0.5778128,
#                 -0.18541728,
#                 -0.46247663,
#             ],
#         ),
#         np.array(-1.5753577),
#     )
#     assert solver.n_vars == 10
#     assert solver.n_cstrs == 4

#     status = solver.dpcd()
#     assert status is 1
#     assert np.isclose(
#         solver.cost, 1.4293642842883392
#     )  # Check if the cost is below the tolerance
#     assert np.allclose(
#         solver.x,
#         np.array(
#             [
#                 0.49443406,
#                 -0.05898912,
#                 -0.16868494,
#                 0.44813251,
#                 -0.16515933,
#                 -0.39468335,
#                 0.86323825,
#                 -0.59236274,
#                 0.52774889,
#                 -0.09063558,
#             ]
#         ),
#     )
#     assert np.all(solver.A @ solver.x - solver.b <= 1e-6)

#     # Add a new constraint to the solver, which makes the problem infeasible
#     solver.add_constraint(-solver.A[0], -solver.b[0] - 0.1)
#     assert solver.n_vars == 10
#     assert solver.n_cstrs == 5

#     # Perform dual coordinate descent
#     status = solver.dpcd()
#     assert status == -1  # Status should be -1 for infeasibility
#     assert np.isposinf(
#         solver.cost
#     )  # Cost should be positive infinity for infeasibility


def test_neuron_init_learning():
    np.random.seed(42)  # For reproducibility

    period = 100.0
    n_f_times = 10
    n_in_channels = 100
    n_in_times = 2000

    f_times = np.sort(np.random.uniform(0.0, period - n_f_times, size=n_f_times))
    f_times += np.arange(n_f_times)

    in_times = np.random.uniform(0.0, period, size=n_in_times)
    in_channels = np.random.randint(0, n_in_channels, size=n_in_times)
    
    neuron = Neuron(n_in_channels)

    neuron.init_learning(
        np.copy(f_times),
        np.copy(in_times),
        np.copy(in_channels),
        period=period,
    )

    assert neuron.weight.shape == (n_in_channels,)
    assert np.all(np.isnan(neuron.weight))
    
    assert neuron.solver is not None
    assert neuron.solver.n_vars == n_in_channels
    assert neuron.solver.n_cstrs == n_f_times


def test_neuron_learn():
    np.random.seed(42)  # For reproducibility

    period = 100.0
    n_f_times = 10
    n_in_channels = 100
    n_in_times = 2000

    f_times = np.sort(np.random.uniform(0.0, period - n_f_times, size=n_f_times))
    f_times += np.arange(n_f_times)

    in_times = np.random.uniform(0.0, period, size=n_in_times)
    in_channels = np.random.randint(0, n_in_channels, size=n_in_times)
    
    neuron = Neuron(n_in_channels)

    neuron.init_learning(
        np.copy(f_times),
        np.copy(in_times),
        np.copy(in_channels),
        period=period,
    )

    res = neuron.learn()
    assert res == 1
    assert np.isclose(neuron.solver.primal_cost, 5.134126071750545)
