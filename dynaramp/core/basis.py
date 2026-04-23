import numpy as np
from scipy.spatial.transform import Rotation


def normalize_vec(vec: np.ndarray) -> np.ndarray:
    """Normalize a vector"""
    try:
        return vec / np.linalg.norm(vec)
    except ZeroDivisionError:
        raise ValueError("Vector cannot be zero vector.")

class Basis:

    """ Frame of reference """

    __global_inst = None

    def __new__(cls, parent:Basis=None):
        if cls.__global_inst is None:
            print("Creating global instance")
            cls.__global_inst = super().__new__(cls)
        if parent is not None:
            return super().__new__(cls)
        return cls.__global_inst

    def __init__(self, parent:Basis=None):
        """3D Basis"""
        self._parent = parent  # global basis if None

        self._mat = np.eye(3)  # rotation matrix relative to parent

    @staticmethod
    def _rotation_matrix(angle:float, axis:np.ndarray) -> np.ndarray:
        """
        Generates a 3x3 rotation matrix using the angle-axis representation.

        This static method calculates the rotation matrix based on a given rotation
        angle and rotation axis. The rotation axis is normalized before performing
        calculations. The provided axis must be a 3-dimensional vector, and the angle
        is applied to create the rotation vector.

        :param angle: The rotation angle in radians.
        :type angle: Float
        :param axis: The axis of rotation, 3-D vector.
        :type axis: np.ndarray
        :return: A 3x3 rotation matrix corresponding to the specified angle and axis.
        :rtype: np.ndarray
        :raises ValueError: If the axis is not a 3-dimensional vector or if the
            axis vector is a zero vector.
        """
        axis = np.asarray(axis, dtype=float)
        try:
            assert axis.shape == (3,)
        except AssertionError:
            raise ValueError("Rotation axis must be a (3,) vector.")

        rot_vec = angle * normalize_vec(axis)
        return Rotation.from_rotvec(rot_vec).as_matrix()

    def _calc_global_transform(self) -> np.ndarray:
        """
        Calculates the global transformation matrix for the current instance.

        This method computes the global transformation of the object by combining its local
        transformation matrix with the parent's global transformation matrix.

        :return: The calculated global transformation matrix as a (3, 3) ndarray.
        :rtype: np.ndarray
        """
        if self._parent is None:
            return self._mat

        parent_mat = self._parent.global_transform
        mat = parent_mat @ self._mat
        # normalize mat uint vectors
        x, y, z = [normalize_vec(u) for u in mat.T]
        mat = np.column_stack((x, y, z))
        return mat

    @property
    def global_transform(self) -> np.ndarray:
        """
        Retrieves the global transformation matrix for the current object.

        This property will always update the global transformation matrix
        before returning it.

        :return: The global transformation matrix.
        :rtype: np.ndarray
        """

        return self._calc_global_transform()

    @global_transform.setter
    def global_transform(self, mat:np.ndarray) -> None:
        """
        Sets the global transform of the object to the specified matrix.

        The provided matrix must be a valid 3x3 rotation matrix.
        Attempting to set the global transform on the global basis will result in an exception.

        :param mat: The new global transform matrix for the basis.
        :type mat: numpy.ndarray
        :raises ValueError: If the provided matrix is not a valid rotation matrix or
            if attempting to set the global transform for the global basis.
        """
        # TODO: Make a real 4x4 transform matrix?
        # Check if mat is a valid rotation matrix.
        try:
            assert isinstance(mat, np.ndarray)
            assert mat.shape == (3, 3)
            # Special orthogonal group check: R.T @ R = I and det(R) = 1
            assert np.allclose(mat.T @ mat, np.eye(3))
            assert np.allclose(np.linalg.det(mat), 1)
        except AssertionError:
            raise ValueError("The given matrix is not a valid rotation matrix.")

        if self._parent is None:
            raise ValueError("Cannot set the global basis.")

        parent_global = self._parent.global_transform
        self._mat = parent_global.T @ mat

    def rotate(self, angle: float, axis: np.ndarray) -> None:
        """
        Rotate the basis around the given axis by the given angle.
        The axis is interpreted in the global frame.

        This is an extrinsic rotation in world coordinates:
            G' = R_world @ G

        Since this basis stores its orientation relative to its parent,
        the global rotation matrix is (intrisic rotation):
            G = P @ L

        Therefore, the equivalent local update is:
            L' = P.T @ R_world @ P @ L

        Where P is the parent's global rotation matrix and L is the current
        local rotation matrix.

        :param angle: Rotation angle in radians.
        :param axis: Rotation axis in global coordinates.
        :return:
        """

        if self._parent is None:
            raise ValueError("Cannot rotate the global basis")

        rot_mat = self._rotation_matrix(angle, axis)
        parent_global = self._parent.global_transform
        self._mat = parent_global.T @ rot_mat @ parent_global @ self._mat


    @property
    def ux(self) -> np.ndarray:
        """
        Retrieves the x unit vector, expressed in global coordinates.

        :return: A 3-D unit vector representing the x-axis.
        :rtype: np.ndarray
        """
        return self.global_transform[:, 0]

    @property
    def uy(self):
        """
        Retrieves the y unit vector, expressed in global coordinates.

        :return: A 3-D unit vector representing the y-axis.
        :rtype: np.ndarray
        """
        return self.global_transform[:, 1]

    @property
    def uz(self):
        """
        Retrieves the z unit vector, expressed in global coordinates.

        :return: A 3-D unit vector representing the z-axis.
        :rtype: np.ndarray
        """
        return self.global_transform[:, 2]

    def map_local_to(self, other, vec):
        return other.global_transform @ self.global_transform.T @ vec
