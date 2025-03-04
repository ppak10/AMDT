from .base import SolverBase
from .models import SolverModels
from .utils import SolverUtils


class Solver(SolverBase, SolverModels, SolverUtils):
    def __init__(
        self,
        build_config_file="default.ini",
        build={},
        material_config_file="SS316L.ini",
        material={},
        mesh_config_file="scale_millimeter.ini",
        mesh={},
        verbose: bool = False,
        **kwargs,
    ):
        """
        @param build_config_file: File for default build parameter values
        @param build: Build parameter overrides
        @param material_config_file: File for default material parameter values
        @param material: Material parameter overrides
        @param mesh_config_file: File for default mesh configuration values,
        @param mesh: Mesh configuration overrides
        @param verbose: For debugging
        """
        super().__init__(
            build_config_file=build_config_file,
            build=build,
            material_config_file=material_config_file,
            material=material,
            mesh_config_file=mesh_config_file,
            mesh=mesh,
            verbose=verbose,
            **kwargs,
        )
