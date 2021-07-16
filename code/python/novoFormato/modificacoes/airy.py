from math import pi, cos, sqrt, sin, cosh, sinh


class Airy:

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

    # Criar todos os cálculos necessários para o estudo de velocidade e aceleração
    def vel_horizontal(self, t, x, z):
        return (((self.g*self.k*self.H)/(2*self.w))*(cosh(self.k*(z+self.d))/cosh(self.k*self.d))*cos(self.k*(x-self.c*t)))

    def vel_vertical(self, t, x, z):
        return (((self.g*self.k*self.H)/(2*self.w))*(sinh(self.k*(z+self.d))/cosh(self.k*self.d))*sin(self.k*(x-self.c*t)))

    def ac_horizontal(self, t, x, z):
        return (((self.g*self.k*self.H)/(2))*(cosh(self.k*(z+self.d))/cosh(self.k*self.d))*sin(self.k*(x-self.c*t)))

    def ac_vertical(self, t, x, z):
        return -(((self.g*self.k*self.H)/(2))*(sinh(self.k*(z+self.d))/cosh(self.k*self.d))*cos(self.k*(x-self.c*t)))
