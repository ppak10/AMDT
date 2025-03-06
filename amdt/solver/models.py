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
    def eagar_tsai(parameters: SolverForwardParameters):
        """
        Provides next state for eagar tsai modeling
        """
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

        # Centering for keeping segment within bounds of x and y
        xs = parameters["xs"] - parameters["xs"][len(parameters["xs"]) // 2]
        ys = parameters["ys"] - parameters["ys"][len(parameters["ys"]) // 2]
        zs = parameters["zs"]

        theta = np.ones((len(xs), len(ys), len(zs))) * parameters["t_0"]

        x_coord = xs[:, None, None, None]
        y_coord = ys[None, :, None, None]
        z_coord = zs[None, None, :, None]

        def integral(tau):
            xp = -parameters["velocity"] * tau * np.cos(parameters["phi"])
            yp = -parameters["velocity"] * tau * np.sin(parameters["phi"])

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

    @staticmethod
    def rosenthal(parameters: SolverForwardParameters):
        """
        Provides next state for rosenthal modeling
        """
        # Thermal Diffusivity (Wolfer et al. Equation 1)
        D = parameters["k"] / (parameters["rho"] * parameters["c_p"])

        # Centering within the x and y plane
        xs = parameters["xs"] - parameters["xs"][len(parameters["xs"]) // 2]
        ys = parameters["ys"] - parameters["ys"][len(parameters["ys"]) // 2]
        zs = parameters["zs"]

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

        theta = np.ones((len(xs), len(ys), len(zs))) * parameters["t_0"]

        coefficient = (
            parameters["absorptivity"]  # A
            * parameters["power"]  # P
            / (2 * np.pi * parameters["k"])
        )

        # TODO: Implement time evolution properly.
        # May not be as visible for smaller lengths of dt however, for longer
        # dt it will show up as less heat.

        # One way to solve this is to make a consistent dt in the gcode segments
        # Thus avoid this issue entirely as the dt provided by the segments is
        # the same everywhere and all are added consistently to theta.

        # Velocity components
        vx = parameters["velocity"] * np.cos(parameters["phi"])
        vy = parameters["velocity"] * np.sin(parameters["phi"])

        # Rotate the frame by phi
        X_rot = X * np.cos(parameters["phi"]) + Y * np.sin(parameters["phi"])
        Y_rot = -X * np.sin(parameters["phi"]) + Y * np.cos(parameters["phi"])

        for t in np.linspace(0, parameters["dt"], num=1):
            # In the moving frame, shift the coordinates relative to the heat source
            x_rel = X_rot + vx * t
            y_rel = Y_rot + vy * t

            # Assuming x is along the weld center line
            zeta = -x_rel

            # r is the cylindrical radius composed of y and z
            r = np.sqrt(y_rel**2 + Z**2)

            R = np.sqrt(zeta**2 + r**2)

            # Rosenthal temperature contribution
            temp_rise = (coefficient / R) * np.exp(
                (parameters["velocity"] * (zeta - R)) / (2 * D)
            )

            temp = np.minimum(temp_rise, 1673)

            # Mask temperatures close to background.
            # temp[temp < 500] = 0
            temp[temp < 1673] = 0

            # Add contribution to the temperature field
            theta += temp

        return theta
