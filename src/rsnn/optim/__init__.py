from collections import defaultdict

import gurobipy as gp

GUROBI_STATUS = defaultdict(lambda: "unknown status")
GUROBI_STATUS[gp.GRB.OPTIMAL] = "optimal"
GUROBI_STATUS[gp.GRB.INFEASIBLE] = "infeasible"
GUROBI_STATUS[gp.GRB.UNBOUNDED] = "unbounded"
GUROBI_STATUS[gp.GRB.INF_OR_UNBD] = "infeasible or unbounded"
GUROBI_STATUS[gp.GRB.NUMERIC] = "numeric error"
