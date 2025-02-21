from pySDC.core.errors import TransferError
from pySDC.core.space_transfer import SpaceTransfer
from pySDC.implementations.datatype_classes.particles import particles, fields, acceleration
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh

class particles_to_particles(SpaceTransfer):
    """
    Custom transfer class, implements SpaceTransfer.py

    This implementation is just a dummy for particles with no direct functionality, i.e. the number of particles is not
    reduced on the coarse problem
    """

    def restrict(self, F):
        """
        Dummy restriction routine

        Args:
            F: the fine level data
        """

        if isinstance(F, particles):
            G = particles(F)
        elif isinstance(F, fields):
            G = fields(F)
        elif isinstance(F, acceleration):
            G = acceleration(F)
        else:
            raise TransferError("Unknown type of fine data, got %s" % type(F))
        return G

    def prolong(self, G):
        """
        Dummy prolongation routine

        Args:
            G: the coarse level data
        """

        if isinstance(G, particles):
            F = particles(G)
        elif isinstance(G, fields):
            F = fields(G)
        elif isinstance(G, acceleration):
            F = acceleration(G)
        else:
            raise TransferError("Unknown type of coarse data, got %s" % type(G))
        return F

class mesh_to_mesh(SpaceTransfer):
    """
    Custom transfer class, implements
    """

    def restrict(self, F):
        """
        Dummy restriction routine

        Args:
            F: the fine level data
        """

        if isinstance(F, mesh):
            G = mesh(F)
        elif isinstance(F, imex_mesh):
            G = imex_mesh(F)
        else:
            raise TransferError("Unknown type of fine data, got %s" % type(F))
        return G

    def prolong(self, G):
        """
        Dummy prolongation routine

        Args:
            G: the coarse level data
        """

        if isinstance(G, mesh):
            F = mesh(G)
        elif isinstance(G, imex_mesh):
            F = imex_mesh(G)
        else:
            raise TransferError("Unknown type of coarse data, got %s" % type(G))
        return F
