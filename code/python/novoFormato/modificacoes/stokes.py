from math import pi, cos, sqrt, sin, cosh, sinh


class Stokes:

    def __init__(self, profundidade, altura, periodo):
        self.d = profundidade
        self.H = altura
        self.T = periodo
        self.g = 9.81
        self.ComputeLength()
        self.k = 2*pi/self.L
        self.w = (2*pi/self.T)
        self.c = (self.w/self.k)

    def ComputeLength(self):
        W = ((4*(pi**2)*self.d)/(self.g*(self.T**2)))
        f = (1 + (0.666*W + 0.445*W**2 - 0.105*W**3 + 0.272*W**4))
        self.L = ((self.T*sqrt(self.g*self.d)*sqrt(f/(1+W*f))))

    def vel_horizontal_ordem1(self, t, z, x):
        return ((((pi*self.H)/(self.T))*(cosh(self.k*(z+self.d))/sinh(self.k*self.d))*cos(self.k*(x-self.c*t))))

    def vel_horizontal_ordem2(self, t, z, x):
        return ((3/4*self.c)*(((pi*self.H)/self.T)**2)*((cosh(2*self.k*(z+self.d)))/((sinh(self.k*self.d))**4))*(cos(2*(self.k*(x-self.c*t)))))

    def vel_horizontal(self, t, z, x):
        u1hl = self.vel_horizontal_ordem1(t, z, x)
        u2hl = self.vel_horizontal_ordem2(t, z, x)
        return u1hl + u2hl

    def vel_vertical_ordem1(self, t, z, x):
        return (((pi*self.H)/(self.T))*(sinh(self.k*(z+self.d))/sinh(self.k*self.d))*sin(self.k*(x-self.c*t)))

    def vel_vertical_ordem2(self, t, z, x):
        return ((3/(4*self.c))*((pi*self.H/self.T)**2)*((sinh(2*self.k*(z+self.d))/(sinh(self.k*self.d)**4)))*(sin(2*self.k*(x - self.c*t))))

    def vel_vertical(self, t, z, x):
        v1hL = self.vel_vertical_ordem1(t, z, x)
        v2hL = self.vel_vertical_ordem2(t, z, x)
        return v1hL + v2hL

    def ac_vertical_ordem1(self, t, z, x):
        return (((2*(pi**2)*self.H)/(self.T**2))*((cosh(self.k*(z+self.d)))/(sinh(self.k*self.d)))*(sin(self.k*(x-self.c*t))))

    def ac_vertical_ordem2(self, t, z, x):
        return ((3*pi/(2*self.L))*(((pi*self.H)/(self.T))**2)*((cosh(2*self.k*(z+self.d))/((sinh(self.k*self.d))**4)))*(sin(2*(x - self.c*t))))

    def ac_vertical(self, t, z, x):
        A1uL = self.ac_vertical_ordem1(t, z, x)
        A2uL = self.ac_vertical_ordem2(t, z, x)
        return A1uL + A2uL

    def ac_horizontal_ordem1(self, t, z, x):
        return (-(((2*(pi**2)*self.H)/(self.T**2))*(sinh(self.k*(z+self.d))/sinh(self.k*self.d))*(cos(x-self.c*t))))

    def ac_horizontal_ordem2(self, t, z, x):
        return (-((3*pi/(2*self.L))*(((pi*self.H)/(self.T))**2)*((sinh(2*self.k*(z+self.d))/((sinh(self.k*self.d))**4)))*(cos(2*(x - self.c*t)))))

    def ac_horizontal(self, t, z, x):
        A1hL = self.ac_horizontal_ordem1(t, z, x)
        A2hL = self.ac_horizontal_ordem2(t, z, x)
        return A1hL + A2hL
