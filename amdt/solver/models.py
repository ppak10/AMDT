import numpy as np

from amdt.solver.types import SolverForwardParameters

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

        dt = parameters["dt"]

        # TODO: Fix 75 magic number
        # Splits prescribed delta time into into minimum increments of 10**-4 s.
        num = max(1, int(dt // 10**-4))

        integration = integrate.fixed_quad(integral, START, dt, n=num)

        return theta + integration[0]

    # @staticmethod
    def rosenthal(self, parameters: SolverForwardParameters):
        """
        Provides next state for rosenthal modeling
        """
        alpha = parameters["absorptivity"]
        dt = parameters["dt"]
        phi = parameters["phi"]
        power = parameters["power"]

        # Thermal Diffusivity (Wolfer et al. Equation 1)
        D = parameters["k"] / (parameters["rho"] * parameters["c_p"])

        # Centering within the x and y plane
        xs = parameters["xs"] - parameters["xs"][len(parameters["xs"]) // 2]
        ys = parameters["ys"] - parameters["ys"][len(parameters["ys"]) // 2]
        zs = parameters["zs"]

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

        theta = np.ones((len(xs), len(ys), len(zs))) * parameters["t_0"]

        coefficient = alpha * power / (2 * np.pi * parameters["k"])

        # Splits prescribed delta time into into minimum increments of 10**-4 s.
        num = max(1, int(dt // 10**-4))
        # num = 1

        # TODO: Look into incorporating the diffusion within the segment.
        # For longer segments, since no heat diffusion is applied, it seems like
        # its a long segment of instantenously heated material.
        # prev_theta = theta
        for t in np.linspace(0, dt, num=num):
            # Adds in the expected distance traveled along global x and y axes.
            xp = -parameters["velocity"] * t * np.cos(phi)
            yp = -parameters["velocity"] * t * np.sin(phi)

            # Assuming x is along the weld center line
            zeta = -(X -xp)

            # r is the cylindrical radius composed of y and z
            r = np.sqrt((Y - yp)**2 + Z**2)

            # Rotate the reference frame for Rosenthal by phi
            # Counterclockwise
            # https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions
            if phi > 0:
                zeta_rot = zeta * np.cos(phi) - r * np.sin(phi)
                r_rot = zeta * np.sin(phi) + r * np.cos(phi)

            # Clockwise
            # https://en.wikipedia.org/wiki/Rotation_matrix#Direction
            else:
                zeta_rot = zeta * np.cos(phi) + r * np.sin(phi)
                r_rot = -zeta * np.sin(phi) + r * np.cos(phi)

            R = np.sqrt(zeta_rot**2 + r_rot**2)

            # Rosenthal temperature contribution
            # `notes/rosenthal/#shape_of_temperature_field``
            temp = (coefficient / R) * np.exp(
                (parameters["velocity"] * (zeta_rot - R)) / (2 * D)
            )

            # Prevents showing temperatures above liquidus
            temp = np.minimum(temp, parameters["t_liquidus"])

            # Mask temperatures close to background to prevent "long tail"
            temp[temp < parameters["t_solidus"]] = 0

            # Add contribution to the temperature field
            theta += temp

            # diffuse theta
            # self.theta = self.diffuse(t, theta)

            # prev_theta_diffused = self.diffuse(t, prev_theta)
            # theta = self.graft(t, phi, theta, prev_theta_diffused)
            # theta = prev_theta_diffused
            # prev_theta = theta

        return theta
