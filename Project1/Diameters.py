import math

class Areas():
    def __init__(self, a: float,b: float,c: float):
        self.a = a
        self.b = b
        self.c = c
        self.a1 = None
        self.a2 = None
        self.a3 = None
        self.initArea()

        self.V = None
        self.initVolume()

        self.P1 = None
        self.P2 = None
        self.P3 = None
        self.initPerimeter()

        self.Dv = self.VolumeDiameter()

        self.Da1 = self.AreaDiameter(self.a1)
        self.Da2 = self.AreaDiameter(self.a2)
        self.Da3 = self.AreaDiameter(self.a3)

        self.Dp1 = self.PerimeterVolume(self.P1)
        self.Dp2 = self.PerimeterVolume(self.P2)
        self.Dp3 = self.PerimeterVolume(self.P3)

    def initArea(self) -> None:
        '''
        Finds areas
        '''
        self.a1 = self.a*self.b
        self.a2 = self.a*self.c
        self.a3 = self.b*self.c

    def initVolume(self) -> None:
        '''
        Finds Volume
        '''
        self.V = self.a*self.b*self.c


    def initPerimeter(self) -> None:
        '''
        Finds the perimeter
        '''
        self.P1 = self.a*2+self.b*2
        self.P2 = self.a*2+self.c*2
        self.P3 = self.b*2+self.c*2


    def VolumeDiameter(self):
        """
        Finds equivalent diameter from geometry volume (V)
        """
        Dv = (6 * self.V / math.pi)**(1/3)
        return Dv
        
    def AreaDiameter(self, A):
        '''
        Finds equivalent diameter from surface area (A)
        '''
        Da = math.sqrt(A/math.pi)
        return Da

    def PerimeterVolume(self, P):
        '''
        Finds equivalent diameter from perimeter (P)
        '''
        Dp = P/math.pi
        return Dp

def ProjectedareaDiameter(Ap: float):
        '''
        Finds equivalent diameter from projected area (Ap)
        '''
        Dpa = math.sqrt(4*Ap/math.pi)
        return Dpa



if __name__ == '__main__':
    Box1 = Areas(5,5,5)
    stop = True
