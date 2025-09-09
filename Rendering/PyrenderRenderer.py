import numpy as np
import pyrender
import trimesh
import os

class EfficientMeshRenderer:
    """
    Efficient renderer that loads the mesh once and updates only the pose for repeated renders.
    """

    def __init__(self, ply_model_path, camera_matrix, width=1280, height=720):
        self.K = camera_matrix
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]


        # Load the mesh once
        tm = trimesh.load(ply_model_path)
        self.mesh = pyrender.Mesh.from_trimesh(tm)

        # Set up the scene once
        self.scene = pyrender.Scene()
        self.mesh_node = self.scene.add(self.mesh, pose=np.eye(4))  # Initial pose

        # Set up the camera once (fixed at origin)
        camera = pyrender.IntrinsicsCamera(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy)
        self.scene.add(camera, pose=np.eye(4))

        # Set up the offscreen renderer once
        self.renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

    def render_mask(self, R, t):
        """
        Update the mesh pose and render the boolean mask.

        Args:
            R (np.ndarray): 3x3 rotation matrix (world-to-camera).
            t (np.ndarray): 3x1 translation vector (world-to-camera).

        Returns:
            np.ndarray: Boolean mask (height x width, dtype=bool).
        """
        ADJUSTMENT = np.array(  # Secondary COB to ensure camera axes follow OpenCV convention after rotation
            [[1., 0., 0., 0.],
             [0., -1., 0., 0],
             [0., 0., -1., 0.],
             [0., 0., 0., 1.]]
        )

        # R_fixed = ROT_ADJUSTMENT @ R

        # Update the mesh node's pose matrix (4x4 homogeneous transform)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t.ravel()
        # pose[1, 3] = -pose[1, 3]
        # pose[2, 3] = -pose[2, 3]
        pose = ADJUSTMENT @ pose

        self.mesh_node.matrix = pose

        # Render depth buffer
        depth = self.renderer.render(self.scene, flags=pyrender.RenderFlags.DEPTH_ONLY)

        # Convert to boolean mask
        mask = depth > 0

        return mask

    def cleanup(self):
        """Clean up resources."""
        self.renderer.delete()


# Example usage
if __name__ == "__main__":
    import cv2

    # Example parameters (replace with yours)
    ply_path = os.path.join('../PLY', 'trocar_fixed_joined_ascii.ply')
    fx, fy = 800.0, 800.0
    cx, cy = 640.0, 360.0
    width, height = 1280, 720

    # Initialize renderer once
    renderer = EfficientMeshRenderer(ply_path, fx, fy, cx, cy, width, height)

    # Example: Render with initial pose
    R = np.eye(3)
    t = np.array([0, 0, 1.0])
    mask1 = renderer.render_mask(R, t)

    # Update pose and render again (efficiently)
    t = np.array([0.1, 0.1, -1.0])  # Shifted
    mask2 = renderer.render_mask(R, t)

    # Optional: Visualize
    cv2.imshow("Mask 1", (mask1.astype(np.uint8) * 255))
    cv2.imshow("Mask 2", (mask2.astype(np.uint8) * 255))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Clean up
    renderer.cleanup()