from typing import Optional

from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage

import numpy as np
import torch


class Heron(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "heron",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.array] = None
    ) -> None:
        """[summary]
        """
        
        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            self._usd_path = "/home/isaac_user/Desktop/heron/heron_description/heron.usd"

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            scale=scale
        )
