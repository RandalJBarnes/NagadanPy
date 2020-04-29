"""
Defines and implements a horizontal, homogeneous, isotropic aquifer class
for organizing, accesing, and converting hydrogeologic properties.

Classes
-------
    Aquifer

Author
------
    Dr. Randal J. Barnes
    Department of Civil, Environmental, and Geo- Engineering
    University of Minnesota

Version
-------
    04 April 2020
"""

import numpy as np


class Aquifer:
    """
    A horizontal, homogeneous, isotropic aquifer.

    Attributes
    ----------
    base : double
        The elevation of the base of the aquifer [m].
    thickness : double
        The thickness of the aquifer [m]. thickness > 0.
    porosity : double
        The porosity of the aquifer [ ]. 0 < porosity < 1.
    conductivity : double
        The hydraulic conductivity of th aquifer [m/d]. conductivity > 0.

    Methods
    -------
    elevation_to_head(elevation)
        Convert elevation to piezometric head.
    head_to_elevation(head)
        Convert piezometric head to elevation.
    head_to_potential(head)
        Convert piezometric head to discharge potential.
    potential_to_head(potential)
        Convert discharge potential to piezometric head.
    elevation_to_potential(elevation)
        Convert elevation to discharge potential.
    discharge_to_velocity(discharge, head)
        Convert veritcally integrated discharge vector to vertically averaged
        seepage velocity vector.
    potential_fosm(elevation_ev, elevation_std)
        Compute the avg and std of the discharge potential from the avg and std
        of the elevation, using a first-order-second-moment approximation.
    """

    def __init__(self, base, thickness, porosity, conductivity):
        """
        Initialize all of the attributes for an Aquifer object.

        Parameters
        ----------
        base : double
            The elevation of the base of the aquifer [m].
        thickness : double
            The thickness of the aquifer [m]. thickness > 0.
        porosity : double
            The porosity of the aquifer [ ]. 0 < porosity < 1.
        conductivity : double
            The hydraulic conductivity of th aquifer [m/d]. conductivity > 0.

        Returns
        -------
        None
        """
        self.base = base
        self.thickness = thickness
        self.porosity = porosity
        self.conductivity = conductivity

    def __repr__(self):
        return "Aquifer({0.base}, {0.thickness}, {0.porosity}, {0.conductivity})".format(self)

    def __str__(self):
        return self.__repr__()

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, thickness):
        if thickness <= 0.0:
            raise ValueError("thickness > 0")
        self._thickness = thickness

    @property
    def porosity(self):
        return self._porosity

    @porosity.setter
    def porosity(self, porosity):
        if porosity <= 0.0 or porosity >= 1.0:
            raise ValueError("0 < porosity < 1")
        self._porosity = porosity

    @property
    def conductivity(self):
        return self._conductivity

    @conductivity.setter
    def conductivity(self, conductivity):
        if conductivity <= 0.0:
            raise ValueError("conductivity > 0")
        self._conductivity = conductivity

    def elevation_to_head(self, elevation):
        """
        Convert static water level elevation to piezometric head.

        Parameters
        ----------
        elevation : double
            The static water level elevation [m].

        Raises
        ------
        ValueError
            elevation >= base

        Returns
        -------
        head : double
            The piezometric head measured from the base of the aquifer [m]. head > 0.
        """
        if elevation < self.base:
            raise ValueError("elevation >= base")
        head = elevation - self.base
        return head

    def head_to_elevation(self, head):
        """
        Convert piezometric head to static water level elevation.

        Parameters
        ----------
        head : double
            The piezometric head measured from the base of the aquifer [m]. head > 0.

        Raises
        ------
        ValueError
            head >= 0.

        Returns
        -------
        elevation : double
            The static water level elevation [m].
        """
        if head < 0.0:
            raise ValueError("head >= 0")
        elevation = head + self.base
        return elevation

    def head_to_potential(self, head):
        """
        Convert piezometric head to discharge potential.

        Parameters
        ----------
        head : double
            The piezometric head measured from the base of the aquifer [m].

        Raises
        ------
        ValueError
            head >= 0

        Returns
        -------
        potential : double
            The discharge potential [m^3/d].
        """
        if head < 0.0:
            raise ValueError("head >= 0")
        elif head < self.thickness:
            potential = 0.5*self.conductivity*head**2
        else:
            potential = (self.conductivity * self.thickness
                         * (head - 0.5*self.thickness))
        return potential

    def potential_to_head(self, potential):
        """
        Convert discharge potential to piezometric head.

        Parameters
        ----------
        potential : double
            The discharge potential [m^3/d].

        Raises
        ------
        ValueError
            potential >= 0

        Returns
        -------
        head : double
            The piezometric head measured from the base of the aquifer [m].
        """
        if potential < 0.0:
            raise ValueError("potential >= 0")
        elif potential < 0.5 * self.conductivity * self.thickness**2:
            head = np.sqrt(2.0 * potential / self.conductivity)
        else:
            head = ((potential + 0.5*self.conductivity*self.thickness**2)
                    / (self.conductivity*self.thickness))
        return head

    def elevation_to_potential(self, elevation):
        """
        Convert static water level elevation to discharge potential.

        Parameters
        ----------
        elevation : double
            The static water level elevation [m].

        Returns
        -------
        potential : double
            The discharge potential [m^3/d].
        """
        head = self.elevation_to_head(elevation)
        potential = self.head_to_potential(head)
        return potential

    def potential_to_elevation(self, potential):
        """
        Convert discharge potential to static water level elevation.

        Parameters
        ----------
        potential : double
            The discharge potential [m^3/d].

        Returns
        -------
        elevation : double
            The static water level elevation [m].
        """
        head = self.potential_to_head(potential)
        elevation = self.head_to_elevation(head)
        return elevation

    def discharge_to_velocity(self, discharge, head):
        """
        Convert veritcally integrated discharge vector to vertically averaged
        seepage velocity vector.

        Parameters
        ----------
        discharge : double
            The vertically integrate discharge vector [m^2/d].
        head : double
            The piezometric head measured from the base of the aquifer [m]. head > 0.

        Raises
        ------
        ValueError
            head >= 0

        Returns
        -------
        velocity : double
            The vertically averaged seepage velocity vector [m/d].
        """
        if head < 0.0:
            raise ValueError("head >= 0")
        elif head > self.thickness:
            velocity = discharge / (self.thickness * self.porosity)
        else:
            velocity = discharge / (head * self.porosity)
        return velocity

    def potential_fosm(self, elevation_ev, elevation_std):
        """
        Compute the avg and std of the discharge potential from the avg and std
        of the elevation, using a first-order-second-moment approximation.

        Parameters
        ----------
        elevation_ev : double
            The expected value of the static water level elevation [m].
        elevation_std : double
            The standard deviation of the static water level elevation [m].

        Returns
        -------
        potential_ev : double
            The expected value of the discharge potential [m^3/d].
        potential_std : double
            The standard deviation of the discharge potential [m^3/d].
        """
        head = self.elevation_to_head(elevation_ev)

        if head < self.thickness:
            potential_ev = (0.5*self.conductivity * (head**2 + elevation_std**2))
            potential_std = self.conductivity * np.fabs(head) * elevation_std
        else:
            potential_ev = (self.conductivity * self.thickness * (head - 0.5*self.thickness))
            potential_std = self.conductivity * self.thickness * elevation_std
        return (potential_ev, potential_std)
