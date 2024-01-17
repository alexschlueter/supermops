import cvxopt
import numpy as np
import scipy.sparse as sp

# cf. eq. (18) in Erling D. Andersen: "On formulating quadratic functions in optimization models"
# URL: http://docs.mosek.com/whitepapers/qmodel.pdf
class SeparableConeSolver:
    def __init__(self, space, Msys, y, l1_target, nonneg=True):
        """Formulate the problem as linear cone problem of the form

            min     <c, x>
            s.t.    Gx+s=h
                    s \in C_0 \times C_1

        where C_0 is a nonnegative orthant and C_1 is a second order cone
        (cf. CVXOPT User's Guide, cvxopt.solvers.conelp) and solve it using
        CVXOPT, optionally with MOSEK backend.


        Arguments:
            space {ReducedLinearMotionSpace} -- discretization for dim reduced problem
            Msys {spmatrix or ndarray} -- sparse or dense system matrix
                for main objective
            y {ndarray} -- right hand side vector for main objective
            l1_target {float} -- value for l1 constraint

        Keyword Arguments:
            nonneg {bool} -- True if constraints should enforce nonnegativity
                (default: {True})
        """
        self.space = space

        if sp.issparse(Msys):
            Msys_cvx = cvxopt.spmatrix(Msys.data, Msys.row, Msys.col)
        else:
            Msys_cvx = cvxopt.matrix(Msys)

        # additionally have absolute value of dofs for signed dofs
        total_dofs = space.total_dofs if nonneg else 2 * space.total_dofs
        print(f"total_dofs {total_dofs}")

        self.c = cvxopt.matrix(0.0, (1 + total_dofs, 1))
        self.c[0] = 1

        if nonneg:
            constr_mat, constr_rhs = space.build_nonneg_sum_constr(l1_target)
        else:
            constr_mat, constr_rhs = space.build_l1_constr(l1_target)

        constr_mat_cvx = cvxopt.spmatrix(constr_mat.data, constr_mat.row,
                                         constr_mat.col)

        col_left = cvxopt.matrix(0.0,
                                 (constr_mat.shape[0] + 1 + Msys.shape[0], 1))
        col_left[constr_mat.shape[0]] = -1

        Msys_rows = cvxopt.sparse([
            [
                cvxopt.spmatrix(
                    0.0,
                    [],
                    [],
                    (Msys.shape[0], constr_mat.shape[1] - Msys.shape[1]),
                )
            ],
            [Msys_cvx],
        ])

        self.G = cvxopt.sparse([
            [col_left],
            [
                constr_mat_cvx,
                cvxopt.matrix(0.0, (1, constr_mat.shape[1])),
                Msys_rows,
            ],
        ])

        self.h = cvxopt.matrix(
            [cvxopt.matrix(constr_rhs), 0,
             cvxopt.matrix(y)])
        self.dims = {
            "l": constr_mat.shape[0],
            "q": [1 + Msys.shape[0]],
            "s": []
        }

    def solve(self, backend="mosek"):
        """Solve the optimization problem.

        Keyword Arguments:
            backend {str} -- mosek or cvxopt (default: {"mosek"})

        Returns:
            tuple -- solutions for mu and nu
        """
        if backend == "cvxopt":
            opt_res = cvxopt.solvers.conelp(self.c,
                                            self.G,
                                            self.h,
                                            dims=self.dims)
            if opt_res["status"] != "optimal":
                print("ERROR: CVXOPT returned non-optimal result")
            sol = np.array(opt_res["x"][-self.space.total_dofs:]).flatten(
            )  # first entries are auxiliary vars
        elif backend == "mosek":
            import cvxopt.msk
            import mosek
            solsta, primal, dual = cvxopt.msk.conelp(self.c,
                                                     self.G,
                                                     self.h,
                                                     dims=self.dims)
            if solsta != mosek.solsta.optimal:
                print("ERROR: MOSEK returned non-optimal result!")
            sol = np.array(primal[-self.space.total_dofs:]).flatten(
            )  # first entries are auxiliary vars
        else:
            raise ValueError("Unknown backend: {}".format(backend))

        return self.space.unpack_sol_vec(sol)
