from wxbtool.phys.constants import StefanBoltzmann


def stefan_boltzmann(t):
    return StefanBoltzmann * t * t * t * t
