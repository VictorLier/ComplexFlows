import math
from Diameters import Areas



class box(Areas):
    def __init__(self, a, b, c, rho, rhoWater):
        super().__init__(a, b, c)
        self.g = 9.82
        self.rho = rho
        self.rhoWater = rhoWater
        #self.GForce = self.V*self.rho*self.g - self.V*self.rhoWater*self.g #Force from different densities.



    def EstimateCD(self, D, U) -> float:

        a = (D/2)**2 * math.pi

        V = 4/3 * math.pi * (D/2)**3

        GForce = V*self.rho*self.g - V*self.rhoWater*self.g

        Cd = 2*GForce/(a*U**2*self.rhoWater)

        return Cd
    

if __name__ == '__main__':
    box1 = box(5,5,5,1030,1000)
    cd = box1.EstimateCD(box1.Dv,1.1)
    print(cd)
