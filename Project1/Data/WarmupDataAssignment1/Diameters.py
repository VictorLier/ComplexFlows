import math

def BoxVolume(a: float,b: float,c: float)
    '''
    Finds volume of box with sidelengths a,b,c
    '''
    V = a*b*c
    return V

def Area(a: float,b: float):
    '''
    Finds area from two (a,b) square lengths
    '''
    A = a*b
    return A

def Perimeter(a: float,b: float):
    '''
    Finds the perimeter from two sides (a,b)
    '''
    P = a*2+b*2
    return P


def VolumeDiameter(V: float):
    """
    Finds equivalent diameter from geometry volume (V)
    """
    Dv = (6 * V / math.pi)** (1/3)
    return Dv
    
def AreaDiameter(A: float):
    '''
    Finds equivalent diameter from surface area (A)
    '''
    Da = math.sqrt(A/math.pi)
    return Da

def ProjectedareaDiameter(Ap: float):
    '''
    Finds equivalent diameter from projected area (Ap)
    '''
    Dpa = math.sqrt(4*Ap/math.pi)
    return Dpa

def PerimeterVolume(P: float):
    '''
    Finds equivalent diameter from perimeter (P)
    '''
    Dp = P/math.pi