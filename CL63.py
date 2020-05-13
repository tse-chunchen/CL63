import numpy as np

class CL63:
    
    def __init__(self, members=1, dt=0.01, rs=None, 
                 the=10, r=28, b=8/3., S=1, k1=10, k2=-11, tau=0.1, c=1, ce=0.08, cz=1, ):
        self.members = members
        self.dt = dt
        if rs is None:
            rs = np.random.RandomState()
        self.rs = rs
        self.x = self.rs.standard_normal(size=(members, 9))
        
        ## Model parameters ##
        self.the = the
        self.r = r
        self.b = b
        self.S = S      # Amplitude scale
        self.k1 = k1    # Offset
        self.k2 = k2    # Offset
        self.tau = tau  # Time scale
        self.c = c      # Coupling strength: tro-ocn for x-y
        self.ce = ce    # Coupling strength: tro-extr for x-y
        self.cz = cz    # Coupling strength: tro-ocn for z
        
    def dxdt(self, forcing):
        xe,ye,ze, xt,yt,zt, xo,yo,zo = self.x.T
        the, r, b, = [self.the, self.r, self.b,]
        S, k1, k2, tau, = [self.S, self.k1, self.k2, self.tau,]
        c, ce, cz, = [self.c, self.ce, self.cz]
        
        dxedt = the*(ye - xe) - ce*(S*xt + k1)
        dyedt = r*xe - ye - xe*ze + ce*(S*yt + k1)
        dzedt = xe*ye - b*ze
        
        dxtdt = the*(yt - xt) - c*(S*xo + k2) - ce*(S*xe + k1)
        dytdt = r*xt - yt - xt*zt + c*(S*yo + k2) + ce*(S*ye + k1)
        dztdt = xt*yt - b*zt + cz*zo
        
        dxodt = tau*the*(yo - xo) - c*(xt + k2)
        dyodt = tau*r*xo - tau*yo - tau*S*xo*zo + c*(yt + k2)
        dzodt = tau*S*xo*yo - tau*b*zo - cz*zt
        
        return np.array([dxedt,dyedt,dzedt,
                         dxtdt,dytdt,dztdt,
                         dxodt,dyodt,dzodt,]).T + forcing #np.tile(forcing, (self.members,1))
    
    #def tlm(self):
    
    def advance(self, forcing=np.zeros(9),):
        h = self.dt
        hh = 0.5*h
        h6 = h/6.
        
        x = self.x
        dxdt1 = self.dxdt(forcing)
        self.x = x + hh*dxdt1
        dxdt2 = self.dxdt(forcing)
        self.x = x + hh*dxdt2
        dxdt = self.dxdt(forcing)
        self.x = x + h*dxdt
        dxdt2 = 2.0*(dxdt2 + dxdt)
        dxdt = self.dxdt(forcing)
        self.x = x + h6*(dxdt1 + dxdt + dxdt2)