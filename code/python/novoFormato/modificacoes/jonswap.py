import math
from math import pi, sin, sqrt, cos
import random


class Jonswap:
    def __init__(self, profundidade, altura, periodo):
        self.d = profundidade
        self.H = altura
        self.Tp = periodo
        self.g = 9.81
        self.ComputeLength()
        self.k = 2*pi/self.L
        self.wp = (2*pi/self.Tp)
        self.c = (self.wp/self.k)
        self.y = 6.4*self.Tp**(-0.491)
        self.e = math.e
        self.pf = random.random()
        self.fi = (self.pf*2*pi)
        self.x = 0
        self.z = 0

    def ComputeLength(self):
        W = ((4*(pi**2)*self.d)/(self.g*(self.Tp**2)))
        f = (1 + (0.666*W + 0.445*W**2 - 0.105*W**3 + 0.272*W**4))
        self.L = ((self.Tp*sqrt(self.g*self.d)*sqrt(f/(1+W*f))))

    def espectro(self, w):
        # definição do fator de forma (sig) para o espectro de jonswap
        if w <= self.wp:
            sig = 0.07
        else:
            sig = 0.09
        return ((5/16)*(self.H**2)*((self.wp**4)/(w**5))*((self.e)**(-1.25*((w/self.wp)**(-4))))*(1-0.287*(math.log(self.y)))*(self.y**((self.e)**(-((w-self.wp)**2)/(2*(sig**2)*(self.wp**2))))))

    def amplitude(self, w, dw):
        S = self.espectro(w)
        return sqrt(2*S*dw)

    def elevacao(self, A, w, t):
        return (A*(sin(w*t-self.k*self.x + self.fi)))

    def vel_horizontal(self, w, t, A):
        return (A*w*((self.e)**(self.k*self.z))*(sin(w*t-self.k*self.x+self.fi)))

    def vel_vertical(self, w, t, A):
        return (A*w*((self.e)**(self.k*self.z))*(cos(w*t-self.k*self.x+self.fi)))

    def ac_horizontal(self, w, t, A):
        return (A*(w**2)*((self.e)**(self.k*self.z))*(cos(w*t-self.k*self.x+self.fi)))

    def ac_vertical(self, w, t, A):
        return ((-1)*A*(w**2)*((self.e)**(self.k*self.z))*(sin(w*t-self.k*self.x+self.fi)))
