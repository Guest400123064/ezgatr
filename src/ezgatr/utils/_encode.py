"""
Interface for encoding and decoding geometric entities to and from Projective Geometric
Algebra (PGA) representation. The classes mainly serve as a name space to cover the
conversion utilities for different geometric entities.
"""
import torch


class Point:
    """Interface for 3D Euclidean point to and from PGA representation.

    In PGA [1]_, a point is represented as the intersection of three planes,
    and represented with tri-vectors `e0ij`.

    TODO: WE MAY WANT TO ADD MORE EXPLANATION OF WHY THERE ARE SIGN FLIPS IN THE
    ENCODING AND DECODING FUNCTIONS.

    Reference
    ---------
    .. [1] `"A Guided Tour to the Plane-Based Geometric Algebra PGA", Leo Dorst,
            <https://geometricalgebra.org/downloads/PGA4CS.pdf>`_
    """

    @staticmethod
    def encode(points):
        """Encode 3D Euclidean points to PGA.

        Parameters
        ----------
        points : torch.Tensor
            3D Euclidean points with shape (..., 3).

        Returns
        -------
        torch.Tensor
            PGA representation of the points with shape (..., 16).
        """
        mvs = torch.zeros(
            *points.shape[:-1], 16, dtype=points.dtype, device=points.device
        )

        mvs[..., 14] = 1.0
        mvs[..., 13] = -points[..., 0]
        mvs[..., 12] = points[..., 1]
        mvs[..., 11] = -points[..., 2]

        return mvs

    @staticmethod
    def decode(mvs):
        """ """
        pass


class Plane:
    """Interface for oriented plane to and from PGA
    """

    @staticmethod
    def encode(normals, positions):
        pass

    @staticmethod
    def decode(mvs):
        pass


class Scalar:
    """Interface for scalar to and from PGA representation."""

    @staticmethod
    def encode(scalars):
        pad = torch.zeros(
            *scalars.shape[:-1], 15, dtype=scalars.dtype, device=scalars.device
        )
        return torch.cat([scalars, pad], dim=-1)

    @staticmethod
    def decode(mvs):
        return mvs[..., [0]]


class Pseudoscalar:
    """Interface for pseudoscalar to and from PGA representation."""

    @staticmethod
    def encode(pseudoscalars):
        pad = torch.zeros(
            *pseudoscalars.shape[:-1],
            15,
            dtype=pseudoscalars.dtype,
            device=pseudoscalars.device
        )
        return torch.cat([pad, pseudoscalars], dim=-1)

    @staticmethod
    def decode(mvs):
        return mvs[..., [15]]


class Reflection:
    """Interface for reflection to and from PGA representation.

    In PGA [1]_, a plane can serve as a reflection operator. Therefore, a reflection
    is encoded exactly as a plane.

    Reference
    ---------
    .. [1] `"A Guided Tour to the Plane-Based Geometric Algebra PGA", Leo Dorst,
            <https://geometricalgebra.org/downloads/PGA4CS.pdf>`_
    """

    @staticmethod
    def encode(normals, positions):
        return Plane.encode(normals, positions)

    @staticmethod
    def decode(mvs):
        return Plane.decode(mvs)


class Rotation:
    """Interface for rotation to and from PGA"""

    @staticmethod
    def encode(quaternions):
        pass

    @staticmethod
    def decode(mvs):
        pass


class Translation:
    """
    Interface for translation to and from PGA
    """

    @staticmethod
    def encode(directions):
        pass

    @staticmethod
    def decode(mvs):
        pass
