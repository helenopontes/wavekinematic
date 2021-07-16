import math
from math import pi, sin, sqrt, cos
import random
import numpy as np
import pandas as pd


class PM:
    def __init__(self, profundidade, altura, periodo, qtdondas, x, z, data_hora):
        self.d = profundidade
        self.H = altura
        self.Tp = periodo
        self.nOndas = qtdondas
        self.x = x
        self.z = z
        self.g = 9.81
        self.wp = (2*pi/self.Tp)
        self.wi = 0.01
        self.wf = 5*self.wp
        self.dw = (self.wf - self.wi)/self.nOndas
        self.w = np.arange(self.wi, self.wf, self.dw)
        self.y = 6.4*self.Tp**(-0.491)
        self.e = math.e
        self.k = np.zeros(self.w.size)
        self.fi = np.zeros(self.w.size)
        self.A = np.zeros(self.w.size)
        self.T = np.zeros(self.w.size)
        self.T = 2*pi/self.w
        self.L = np.zeros(self.w.size)
        self.pm = np.zeros(self.w.size)
        j = 0
        for i in self.w:
            self.pm[j] = self.espectro(i)
            self.fi[j] = (random.random()*(2*pi))
            self.A[j] = self.amplitude(i, 0.01)
            W = ((4*(pi**2)*self.d)/(self.g*((1/i)**2)))
            f = (1 + (0.666*W + 0.445*W ** 2 - 0.105*W**3 + 0.272*W**4))
            self.L[j] = ((self.T[j]*sqrt(self.g*self.d)*sqrt(f/(1+W*f))))
            #self.k[j] = (2*pi/self.L[j])
            self.k[j] = self.w[j]*self.w[j]/self.g
            j = j+1

        # Guardando dados no excel
        espectro_valores = {'Frequência': self.w, 'Energia': self.pm}
        excel_espectro = pd.DataFrame(espectro_valores)
        
        #Colocando Data e Hora no nome do arquivo
        #data_hora = datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        excel_espectro_pm = "espectro_pm"
        excel_extensao = ".xlsx"
        excel_espectro.to_excel(excel_espectro_pm + data_hora + excel_extensao)        
        
        #excel_espectro.to_excel('espectro_pm.xlsx')

        # Guardando dados no excel - informações das ondas
        espectro_ondas_valores = {'Frequência': self.w, 'Período': self.T,
                                  'Amplitude': self.A, 'FaseInicial': self.fi, 'Comprimento': self.L}
        excel_espectro_ondas = pd.DataFrame(espectro_ondas_valores)
                
        #Colocando Data e Hora no nome do arquivo
        #data_hora = datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        excel_espectro_ondas_pm = "espectro_ondas_pm"
        excel_extensao = ".xlsx"
        excel_espectro_ondas.to_excel(excel_espectro_ondas_pm + data_hora + excel_extensao)    
        
        #excel_espectro_ondas.to_excel('espectro_ondas_pm.xlsx')

    def espectro(self, w):
        return ((5/16)*(self.H**2)*((self.wp**4)/(w**5))*((self.e)**(-1.25*((w/self.wp)**(-4)))))

    def amplitude(self, w, dw):
        S = self.espectro(w)
        return sqrt(2*S*dw)

    def momento_espectral(self, w):
        S = self.espectro(w)
        return S

    def momento_espectral_2(self, w):
        S = self.espectro(w)
        return S*w**2

    def momento_espectral_4(self, w):
        S = self.espectro(w)
        return S*w**4

    def elevacao(self, w, t, A, nwave):
        #return (A*(sin(w*t-self.k[nwave]*self.x + self.fi[nwave]))) - Modificação
        return (A*(cos(self.k[nwave]*self.x - w*t + self.fi[nwave])))

    def vel_horizontal(self, w, t, A, nwave, z):
        #return (A*w*((self.e)**(self.k[nwave]*self.z))*(sin(w*t-self.k[nwave]*self.x + self.fi[nwave])))
        return (A*w*((self.e)**(self.k[nwave]*z))*(cos(self.k[nwave]*self.x - w*t + self.fi[nwave])))

    def vel_vertical(self, w, t, A, nwave, z):
        #return (A*w*((self.e)**(self.k[nwave]*self.z))*(cos(w*t-self.k[nwave]*self.x+self.fi[nwave])))
        return (A*w*((self.e)**(self.k[nwave]*z))*(sin(self.k[nwave]*self.x - w*t + self.fi[nwave])))

    def ac_horizontal(self, w, t, A, nwave, z):
        #return (A*(w**2)*((self.e)**(self.k[nwave]*self.z))*(cos(w*t-self.k[nwave]*self.x+self.fi[nwave])))
        return (A*(w**2)*((self.e)**(self.k[nwave]*z))*(sin(self.k[nwave]*self.x - w*t + self.fi[nwave])))

    def ac_vertical(self, w, t, A, nwave, z):
        #return ((-1)*A*(w**2)*((self.e)**(self.k[nwave]*self.z))*(sin(w*t-self.k[nwave]*self.x+self.fi[nwave])))
        return ((-1)*A*(w**2)*((self.e)**(self.k[nwave]*z))*(cos(self.k[nwave]*self.x - w*t + self.fi[nwave])))
