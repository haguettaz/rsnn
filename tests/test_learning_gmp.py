import numpy as np

from rsnn.optim.gmp import GMPModel


def test_solver_initialization():
    # Initialize the solver
    solver = GMPModel(42)
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


def test_solver_solve():
    np.random.seed(42)  # For reproducibility

    # Initialize the solver
    solver = GMPModel(10)
    assert solver.n_vars == 10
    assert solver.n_cstrs == 0

    # Solve unconstrained problem
    status = solver.solve()
    assert status == 1  # optimal
    assert np.allclose(solver.x_value(), np.zeros(10))
    assert np.isclose(solver.cost_value(), 0.0)

    # Create a random feasible problem
    A = np.random.randn(4, 10)
    b = A @ np.random.randn(10) + np.random.rand(4)
    solver.add_constraints(A, b)
    assert solver.n_cstrs == 4
    status = solver.solve()
    assert status == 1  # optimal
    assert np.isclose(solver.cost_value(), 0.25631439311777243)

    # Add a few more feasible random constraints to the solver
    A = np.random.randn(2, 10)
    b = A @ np.random.randn(10) + np.random.rand(2)
    solver.add_constraints(A, b)
    assert solver.n_cstrs == 6
    status = solver.solve()
    assert status == 1  # optimal
    assert np.isclose(solver.cost_value(), 0.89746396022)

    # Add a trivial constraint that does not change the solution
    A = np.zeros((1, 10))
    b = np.random.rand(1)
    solver.add_constraints(A, b)
    assert solver.n_cstrs == 7
    status = solver.solve()
    assert status == 1  # optimal
    assert np.isclose(solver.cost_value(), 0.89746396022)

    # Add a few more feasible random constraints to the solver
    A = np.random.randn(1, 10)
    b = A @ np.random.randn(10) + np.random.rand(1)
    solver.add_constraints(A, b)
    assert solver.n_cstrs == 8
    status = solver.solve()
    assert status == 1  # optimal
    assert np.isclose(solver.cost_value(), 0.89746396022)

    # Add an additional constraint that makes the problem infeasible
    solver.add_constraints(-A, -b - np.random.rand(1))
    assert solver.n_cstrs == 9
