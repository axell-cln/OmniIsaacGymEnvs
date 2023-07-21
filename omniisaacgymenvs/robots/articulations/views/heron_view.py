from typing import Optional
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class HeronView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "HeronView"
    ) -> None:
        """[summary]
        """

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=False
        )

        #self.thrusters = [RigidPrimView(prim_paths_expr=f"/World/envs/.*/heron/thruster_{i}", name=f"thruster_{i}_view", reset_xform_properties=False) for i in range(2)]
        self._thrusters_left= RigidPrimView(prim_paths_expr="/World/envs/.*/heron/thruster_1", name="left_thruster_view", reset_xform_properties=False)
        self._thrusters_right= RigidPrimView(prim_paths_expr="/World/envs/.*/heron/thruster_0", name="right_thruster_view", reset_xform_properties=False)
        self._body= RigidPrimView(prim_paths_expr="/World/envs/.*/heron/dummy_link", name="body_view", reset_xform_properties=False)