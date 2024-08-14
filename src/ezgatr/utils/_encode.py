import torch


class Point:
    """
    Interface for 3D point to and from PGA
    """

    @staticmethod
    def encode(points):
        pass

    @staticmethod
    def decode(mvs):
        pass


class Plane:
    """
    Interface for oriented plane to and from PGA
    """

    @staticmethod
    def encode(normals, positions):
        pass

    @staticmethod
    def decode(mvs):
        pass


class Scalar:
    """
    Interface for scalar to and from PGA
    """

    @staticmethod
    def encode(scalars):
        pass

    @staticmethod
    def decode(mvs):
        pass


class Pseudoscalar:
    """
    Interface for pseudoscalar to and from PGA
    """

    @staticmethod
    def encode(pseudoscalars):
        pass

    @staticmethod
    def decode(mvs):
        pass


class Reflection:
    """
    Interface for reflection to and from PGA
    """

    @staticmethod
    def encode(normals, positions):
        return Plane.encode(normals, positions)

    @staticmethod
    def decode(mvs):
        return Plane.decode(mvs)


class Rotation:
    """
    Interface for rotation to and from PGA
    """

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
