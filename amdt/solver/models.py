import numpy as np

from amdt.solver.types import SolverForwardParameters, SolverForwardState

from scipy import integrate

# Small non-zero start for integration
START = 10**-7


class SolverModels:
    """
    Class for solver models such as Eagar-Tsai and Rosenthal.
    """

    @staticmethod
    def eagar_tsai(parameters: SolverForwardParameters, state: SolverForwardState):
        sigma = parameters["beam_diameter"] / 4

        # Coefficient for Equation 16 in Wolfer et al.
        coefficient = (
            parameters["absorptivity"]  # A
            * parameters["power"]  # P
            / (
                2
                * np.pi
                * sigma**2
                * parameters["rho"]
                * parameters["c_p"]
                * np.pi ** (3 / 2)
            )
        )

        # Thermal Diffusivity (Wolfer et al. Equation 1)
        D = parameters["k"] / (parameters["rho"] * parameters["c_p"])

        # Centering, not sure exactly why its like this.
        xs = parameters["xs"] - parameters["xs"][len(parameters["xs"]) // 2]
        ys = parameters["ys"] - parameters["ys"][len(parameters["ys"]) // 2]
        zs = parameters["zs"]

        theta = np.ones((len(xs), len(ys), len(parameters["zs"]))) * parameters["t_0"]

        x_coord = xs[:, None, None, None]
        y_coord = ys[None, :, None, None]
        z_coord = zs[None, None, :, None]

        phi = parameters["phi"]

        def integral(tau):
            xp = -parameters["velocity"] * tau * np.cos(phi)
            yp = -parameters["velocity"] * tau * np.sin(phi)

            lmbda = np.sqrt(4 * D * tau)
            gamma = np.sqrt(2 * sigma**2 + lmbda**2)
            start = (4 * D * tau) ** (-3 / 2)

            # Wolfer et al. Equation A.3
            termy = sigma * lmbda * np.sqrt(2 * np.pi) / (gamma)
            yexp1 = np.exp(-1 * ((y_coord - yp) ** 2) / gamma**2)
            yintegral = termy * (yexp1)

            # Wolfer et al. Equation A.2
            termx = termy
            xexp1 = np.exp(-1 * ((x_coord - xp) ** 2) / gamma**2)
            xintegral = termx * xexp1

            # Wolfer et al. Equation 18
            zintegral = 2 * np.exp(-(z_coord**2) / (4 * D * tau))

            # Wolfer et al. Equation 16
            value = coefficient * start * xintegral * yintegral * zintegral

            return value

        integration = integrate.fixed_quad(integral, START, parameters["dt"], n=75)

        return theta + integration[0]
