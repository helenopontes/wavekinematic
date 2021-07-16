import PySimpleGUI as sg
import math
from math import pi, cos, sqrt, sin, cosh, sinh
import numpy as np
import random
import matplotlib.pyplot as plt
from airy import Airy
from stokes import Stokes
from pierson_moskowitz import PM
from jonswap import Jonswap

# Criar janelas e layouts


def inicio():
    layout = [
        [sg.Text('Qual o tipo de estudo que deseja fazer?')],
        [sg.Radio('Ondas Regulares', 'tipo', key='regular', default=True), sg.Radio(
            'Ondas Irregulares', 'tipo', key='irregular')],
        [sg.Button('Continuar')],
    ]
    return sg.Window(
        'Wave Kinematics', layout=layout, finalize=True)


def regular():
    layout = [
        [sg.Text('Insira o valor da altura da onda:'),
         sg.Input(size=(15, 0), key='altura')],
        [sg.Text('Insira o valor do período da onda:'),
         sg.Input(size=(15, 0), key='periodo')],
        [sg.Text('Insira o valor da profundidade da água:'),
         sg.Input(size=(15, 0), key='profundidade')],
        [sg.Text('Insira o valor da posição de onda:'),
         sg.Input(size=(15, 0), key='posicao')],
        [sg.Text('Insira o valor de tempo:'),
         sg.Input(size=(15, 0), key='tempo')],
        [sg.Text('Qual teoria de onda será usada?')],
        [sg.Radio('Airy', 'teoria', key='airy', default='True'),
         sg.Radio('Stokes', 'teoria', key='stokes')],
        [sg.Text('Deseja analisar as propriedades ao longo:')],
        [sg.Radio('do Tempo', 'tipo', key='choice_tempo', default='True'), sg.Radio(
            'da Profundidade', 'tipo', key='choice_profundidade')],
        [sg.Text('Assinale quais propriedades serão analisadas')],
        [sg.Checkbox('Velocidade Horizontal', key='vel_horizontal'), sg.Checkbox(
            'Velocidade Vertical', key='vel_vertical')],
        [sg.Checkbox('Aceleração Horizontal', key='ac_horizontal'), sg.Checkbox(
            'Aceleração Vertical', key='ac_vertical')],
        [sg.Button('Voltar'), sg.Button('Processar Dados')],
        [sg.Text('Resultados:')],
        # [sg.Output()]
    ]
    # janela
    return sg.Window(
        'Wave Kinematics - Ondas Regulares', layout=layout, finalize=True)


def irregular():
    layout = [
        [sg.Text('Insira o valor da altura significativa da onda:'),
         sg.Input(size=(15, 0), key='altura')],
        [sg.Text('Insira o valor do período de pico da onda:'),
         sg.Input(size=(15, 0), key='periodo_pico')],
        [sg.Text('Insira o valor da profundidade da água:'),
         sg.Input(size=(15, 0), key='profundidade')],
        [sg.Text('Qual tipo de espectro de onda será analisado:')],
        [sg.Radio('Pierson-Moskowitz', 'espectro', key='pierson_moskowitz', default=True),
         sg.Radio('Jonswap', 'espectro', key='jonswap')],
        [sg.Text('Assinale quais propriedades serão analisadas')],
        [sg.Checkbox('Espectro de Energia', key='espectro'), sg.Checkbox(
            'Elevacao de Superfície', key='elevacao')],
        [sg.Checkbox('Velocidade Horizontal', key='vel_horizontal'), sg.Checkbox(
            'Velocidade Vertical', key='vel_vertical')],
        [sg.Checkbox('Aceleração Horizontal', key='ac_horizontal'), sg.Checkbox(
            'Aceleração Vertical', key='ac_vertical')],
        [sg.Button('Voltar'), sg.Button('Processar Dados')],
    ]
    return sg.Window(
        'Wave Kinematics - Ondas Irregulares', layout=layout, finalize=True)


# Criar janelas iniciais
janela1, janela2, janela3 = inicio(), None, None

# Criar Loop de eventos
while True:
    window, event, values = sg.read_all_windows()
    # fechar janela
    if window == janela1 and event == sg.WIN_CLOSED:
        break
        # Próxima Janela
    if window == janela1 and event == 'Continuar':
        if values['regular'] == True:
            janela2 = regular()
            janela1.hide()
        elif values['irregular'] == True:
            janela3 = irregular()
            janela1.hide()

    # fechar janela
    elif window == janela2 and event == sg.WIN_CLOSED:
        break

    elif window == janela2 and event == 'Voltar':
        janela2.hide()
        janela1.un_hide()

    elif window == janela2 and event == 'Processar Dados':
        # Extrair dados da tela
        button, values = janela2.Read()
        H = float(values['altura'])
        T = float(values['periodo'])
        d = float(values['profundidade'])
        x = float(values['posicao'])
        t = float(values['tempo'])
        airy = values['airy']
        stokes = values['stokes']
        vel_horizontal = values['vel_horizontal']
        vel_vertical = values['vel_vertical']
        ac_horizontal = values['ac_horizontal']
        ac_vertical = values['ac_vertical']
        choice_tempo = values['choice_tempo']
        choice_profundidade = values['choice_profundidade']

        if airy == True and stokes == False:
            # Calculo percentual das velocidades e acelerações
            # Nomear variaveis de acordo com cada função criada, após isso criar as variáveis percentuais
            wave1 = Airy(d, H, T)
            uhL0 = wave1.vel_horizontal(t, x, 0.0)
            uhL2 = wave1.vel_horizontal(t, x, -wave1.L/2)
            uvL0 = wave1.vel_vertical(t, x, 0.0)
            uvL2 = wave1.vel_vertical(t, x, -wave1.L/2)
            AvL0 = wave1.ac_vertical(t, x, 0.0)
            AvL2 = wave1.ac_vertical(t, x, -wave1.L/2)
            AhL0 = wave1.ac_horizontal(t, x, 0.0)
            AhL2 = wave1.ac_horizontal(t, x, -wave1.L/2)

            PVh = 100*(uhL2/uhL0)
            PVv = 100*(uvL2/uvL0)
            PAv = 100*(AvL2/AvL0)
            PAh = 100*(AhL2/AhL0)
            print(
                f'Variáveis percentuais: \nVelocidade Horizontal: {PVh} \nVelocidade Vertical: {PVv} \nAceleração Horizontal: {PAh} \nAceleração Vertical: {PAv}')

            # Criação de gráficos

            if choice_tempo == True and choice_profundidade == False:
                # Plotando ao longo do tempo

                t1 = np.arange(0, 100, 1)
                vel_horizontal_t = np.zeros(100)
                vel_vertical_t = np.zeros(100)
                ac_horizontal_t = np.zeros(100)
                ac_vertical_t = np.zeros(100)
                for i in t1:
                    vel_horizontal_t[i] = wave1.vel_horizontal(i, x, 0.0)
                    vel_vertical_t[i] = wave1.vel_vertical(i, x, 0.0)
                    ac_horizontal_t[i] = wave1.ac_horizontal(i, x, 0.0)
                    ac_vertical_t[i] = wave1.ac_vertical(i, x, 0.0)
                plt.clf()
                plt.title('Teoria de Airy')
                if vel_horizontal == True:
                    plt.plot(t1, vel_horizontal_t, 'b',
                             label='Velocidade Horizontal')
                    plt.xlabel('Tempo')
                    plt.ylabel('Velocidade')
                else:
                    pass
                if vel_vertical == True:
                    plt.plot(t1, vel_vertical_t, 'g',
                             label='Velocidade Vertical')
                    plt.xlabel('Tempo')
                    plt.ylabel('Velocidade')
                else:
                    pass
                if ac_horizontal == True:
                    plt.plot(t1, ac_horizontal_t, 'r',
                             label='Aceleração Horizontal')
                    plt.xlabel('Tempo')
                    plt.ylabel('Aceleração')
                else:
                    pass
                if ac_vertical == True:
                    plt.plot(t1, ac_vertical_t, 'lime',
                             label='Aceleração Vertical')
                    plt.xlabel('Tempo')
                    plt.ylabel('Aceleração')
                else:
                    pass
                plt.legend()
                plt.grid()
                plt.show()
            else:
                # Plotando ao longo da profundidade

                z = int(d)
                z1 = np.arange(0, z, 1)
                vel_horizontal_z = np.zeros(z)
                vel_vertical_z = np.zeros(z)
                ac_horizontal_z = np.zeros(z)
                ac_vertical_z = np.zeros(z)
                for i in z1:
                    vel_horizontal_z[i] = wave1.vel_horizontal(t, x, i)
                    vel_vertical_z[i] = wave1.vel_vertical(t, x, i)
                    ac_horizontal_z[i] = wave1.ac_horizontal(t, x, i)
                    ac_vertical_z[i] = wave1.ac_vertical(t, x, i)
                plt.clf()
                plt.title('Teoria de Airy')
                if vel_horizontal == True:
                    plt.plot(z1, vel_horizontal_z, 'b',
                             label='Velocidade Horizontal')
                    plt.xlabel('Profundidade')
                    plt.ylabel('Velocidade')
                else:
                    pass
                if vel_vertical == True:
                    plt.plot(z1, vel_vertical_z, 'g',
                             label='Velocidade Vertical')
                    plt.xlabel('Profundidade')
                    plt.ylabel('Velocidade')
                else:
                    pass
                if ac_horizontal == True:
                    plt.plot(z1, ac_horizontal_z, 'r',
                             label='Aceleração Horizontal')
                    plt.xlabel('Profundidade')
                    plt.ylabel('Aceleração')
                else:
                    pass
                if ac_vertical == True:
                    plt.plot(z1, ac_vertical_z, 'lime',
                             label='Aceleração Vertical')
                    plt.xlabel('Profundidade')
                    plt.ylabel('Aceleração')
                else:
                    pass
                plt.legend()
                plt.grid()
                plt.show()

        elif airy == False and stokes == True:
            # Calculo percentual das velocidades e acelerações
            # Nomear variaveis de acordo com cada função criada, após isso criar as variáveis percentuais
            wave2 = Stokes(d, H, T)
            uthL0 = wave2.vel_horizontal(t, x, 0.0)
            uthL2 = wave2.vel_horizontal(t, x, -wave2.L/2)
            vthL0 = wave2.vel_vertical(t, x, 0.0)
            vthL2 = wave2.vel_vertical(t, x, -wave2.L/2)
            AtuL0 = wave2.ac_vertical(t, x, 0.0)
            AtuL2 = wave2.ac_vertical(t, x, -wave2.L/2)
            AthL0 = wave2.ac_horizontal(t, x, 0.0)
            AthL2 = wave2.ac_horizontal(t, x, -wave2.L/2)

            PsVh = 100*(uthL2/uthL0)
            PsVv = 100*(vthL2/vthL0)
            PsAv = 100*(AtuL2/AtuL0)
            PsAh = 100*(AthL2/AthL0)
            print(
                f'Variáveis percentuais: \nVelocidade Horizontal: {PsVh} \n Velocidade Vertical: {PsVv} \n Aceleração Horizontal: {PsAh} \n Aceleração Vertical: {PsAv} ')

            # Plotando o gráficos
            if choice_tempo == True and choice_profundidade == False:
                # Plotando ao longo do tempo
                t1 = np.arange(0, 100, 1)
                vel_horizontal_t = np.zeros(100)
                vel_vertical_t = np.zeros(100)
                ac_horizontal_t = np.zeros(100)
                ac_vertical_t = np.zeros(100)
                for i in t1:
                    vel_horizontal_t[i] = wave2.vel_horizontal(i, x, 0.0)
                    vel_vertical_t[i] = wave2.vel_vertical(i, x, 0.0)
                    ac_horizontal_t[i] = wave2.ac_horizontal(i, x, 0.0)
                    ac_vertical_t[i] = wave2.ac_vertical(i, x, 0.0)
                plt.clf()
                plt.title('Teria de Stokes')
                if vel_horizontal == True:
                    plt.plot(t1, vel_horizontal_t, 'b',
                             label='Velocidade Horizontal')
                    plt.xlabel('Tempo')
                    plt.ylabel('Velocidade')
                else:
                    pass
                if vel_vertical == True:
                    plt.plot(t1, vel_vertical_t, 'g',
                             label='Velocidade Vertical')
                    plt.xlabel('Tempo')
                    plt.ylabel('Velocidade')
                else:
                    pass
                if ac_horizontal == True:
                    plt.plot(t1, ac_horizontal_t, 'r',
                             label='Aceleração Horizontal')
                    plt.xlabel('Tempo')
                    plt.ylabel('Aceleração')
                else:
                    pass
                if ac_vertical == True:
                    plt.plot(t1, ac_vertical_t, 'lime',
                             label='Aceleração Vertical')
                    plt.xlabel('Tempo')
                    plt.ylabel('Aceleração')
                else:
                    pass
                plt.legend()
                plt.grid()
                plt.show()

            else:
                # Plotando ao longo da profundidade
                z = int(d)
                z1 = np.arange(0, z, 1)
                vel_horizontal_z = np.zeros(z)
                vel_vertical_z = np.zeros(z)
                ac_horizontal_z = np.zeros(z)
                ac_vertical_z = np.zeros(z)
                for i in z1:
                    vel_horizontal_z[i] = wave2.vel_horizontal(t, x, -i)
                    vel_vertical_z[i] = wave2.vel_vertical(t, x, -i)
                    ac_horizontal_z[i] = wave2.ac_horizontal(t, x, -i)
                    ac_vertical_z[i] = wave2.ac_vertical(t, x, -i)

                plt.clf()
                plt.title('Teria de Stokes')
                if vel_horizontal == True:
                    plt.plot(z1, vel_horizontal_z, 'b',
                             label='Velocidade Horizontal')
                    plt.xlabel('Profundidade')
                    plt.ylabel('Velocidade')
                else:
                    pass
                if vel_vertical == True:
                    plt.plot(z1, vel_vertical_z, 'g',
                             label='Velocidade Vertical')
                    plt.xlabel('Profundidade')
                    plt.ylabel('Velocidade')
                else:
                    pass
                if ac_horizontal == True:
                    plt.plot(z1, ac_horizontal_z, 'r',
                             label='Aceleração Horizontal')
                    plt.xlabel('Profundidade')
                    plt.ylabel('Aceleração')
                else:
                    pass
                if ac_vertical == True:
                    plt.plot(z1, ac_vertical_z, 'lime',
                             label='Aceleração Vertical')
                    plt.xlabel('Profundidade')
                    plt.ylabel('Aceleração')
                else:
                    pass
                plt.legend()
                plt.grid()
                plt.show()

    elif window == janela3 and event == sg.WIN_CLOSED:
        break

    elif window == janela3 and event == 'Voltar':
        janela3.hide()
        janela1.un_hide()

    elif window == janela3 and event == 'Processar Dados':
        button, values = janela3.Read()
        H = float(values['altura'])
        Tp = float(values['periodo_pico'])
        d = float(values['profundidade'])
        choice_pm = values['pierson_moskowitz']
        choice_jonswap = values['jonswap']
        espectro = values['espectro']
        elevacao = values['elevacao']
        vel_horizontal = values['vel_horizontal']
        vel_vertical = values['vel_vertical']
        ac_horizontal = values['ac_horizontal']
        ac_vertical = values['ac_vertical']

        if choice_pm == True and choice_jonswap == False:
            # Gráfico do Espectro de onda x frequencia
            pierson_moskowitz = PM(d, H, Tp)
            wf = 5*(pierson_moskowitz.wp)
            dw = np.arange(0.01, wf, 0.01)
            pm = np.zeros(dw.size)
            fi = np.zeros(dw.size)
            amplitude_pm = np.zeros(dw.size)
            k = np.zeros(dw.size)
            j = 0
            for i in dw:
                pm[j] = pierson_moskowitz.espectro(i)
                fi[j] = (random.random()*(2*pi))
                amplitude_pm[j] = pierson_moskowitz.amplitude(i, 0.01)
                W = ((4*(pi**2)*d)/(9.81*((1/i)**2)))
                f = (1 + (0.666*W + 0.445*W **
                          2 - 0.105*W**3 + 0.272*W**4))
                L = ((Tp*sqrt(9.81*d)*sqrt(f/(1+W*f))))
                k[j] = (2*pi/L)
                j = j+1

            # elevação, velocidades e acelerações
            t1 = np.arange(0, 400, 1)
            pm_elevacao = np.zeros(t1.size)
            pm_vel_horizontal = np.zeros(t1.size)
            pm_vel_vertical = np.zeros(t1.size)
            pm_ac_horizontal = np.zeros(t1.size)
            pm_ac_vertical = np.zeros(t1.size)
            j = 0
            for l in t1:
                nwave = 0
                for i in dw:
                    pm_elevacao[j] += pierson_moskowitz.elevacao(
                        i, l, amplitude_pm[nwave])
                    pm_vel_horizontal[j] += pierson_moskowitz.vel_horizontal(
                        i, l, amplitude_pm[nwave])
                    pm_vel_vertical[j] += pierson_moskowitz.vel_vertical(
                        i, l, amplitude_pm[nwave])
                    pm_ac_horizontal[j] += pierson_moskowitz.ac_horizontal(
                        i, l, amplitude_pm[nwave])
                    pm_ac_vertical[j] += pierson_moskowitz.ac_vertical(
                        i, l, amplitude_pm[nwave])
                    nwave = nwave + 1
                j = j + 1
            # plotagem dos gráficos
            # espectro de energia
            plt.clf()
            plt.title('Pierson-Moskowitz')
            if espectro == True:
                plt.plot(dw, pm, 'blueviolet',
                         label='Espectro de Onda')
                plt.xlabel('Frequência')
                plt.ylabel('Energia')
            else:
                pass

            # elevacao, velocidades e acelerações
            if elevacao == True:
                plt.plot(t1, pm_elevacao, 'orange',
                         label='Elevação de Superfíce')
                plt.xlabel('Tempo')
                plt.ylabel('Elevação')
            else:
                pass
            if vel_horizontal == True:
                plt.plot(t1, pm_vel_horizontal, 'b',
                         label='Velocidade Horizontal')
                plt.xlabel('Tempo')
                plt.ylabel('Velocidade')
            else:
                pass
            if vel_vertical == True:
                plt.plot(t1, pm_vel_vertical, 'g',
                         label='Velocidade Vertical')
                plt.xlabel('Tempo')
                plt.ylabel('Velocidade')
            else:
                pass
            if ac_horizontal == True:
                plt.plot(t1, pm_ac_horizontal, 'r',
                         label='Aceleração Horizontal')
                plt.xlabel('Tempo')
                plt.ylabel('Aceleração')
            else:
                pass
            if ac_vertical == True:
                plt.plot(t1, pm_ac_vertical, 'lime',
                         label='Aceleração Vertical')
                plt.xlabel('Tempo')
                plt.ylabel('Aceleração')
            else:
                pass
            plt.legend()
            plt.grid()
            plt.show()

        else:
            # Gráfico do Espectro de onda x frequencia
            jonswap = Jonswap(d, H, Tp)
            wf = 5*(jonswap.wp)
            dw = np.arange(0.01, wf, 0.01)
            jp = np.zeros(dw.size)
            fi = np.zeros(dw.size)
            amplitude_j = np.zeros(dw.size)
            k = np.zeros(dw.size)
            j = 0
            for i in dw:
                jp[j] = jonswap.espectro(i)
                fi[j] = (random.random()*(2*pi))
                amplitude_j[j] = jonswap.amplitude(i, 0.01)
                W = ((4*(pi**2)*d)/(jonswap.g*((1/i)**2)))
                f = (1 + (0.666*W + 0.445*W **
                          2 - 0.105*W**3 + 0.272*W**4))
                L = ((Tp*sqrt(jonswap.g*d)*sqrt(f/(1+W*f))))
                k[j] = (2*pi/L)
                j = j+1
                # elevação, velocidades e acelerações
            t1 = np.arange(0, 400, 1)
            jonswap_elevacao = np.zeros(t1.size)
            jonswap_vel_horizontal = np.zeros(t1.size)
            jonswap_vel_vertical = np.zeros(t1.size)
            jonswap_ac_horizontal = np.zeros(t1.size)
            jonswap_ac_vertical = np.zeros(t1.size)
            j = 0
            for l in t1:
                nwave = 0
                for i in dw:
                    jonswap_elevacao[j] += jonswap.elevacao(
                        i, l, amplitude_pm[nwave])
                    jonswap_vel_horizontal[j] += jonswap.vel_horizontal(
                        i, l, amplitude_pm[nwave])
                    jonswap_vel_vertical[j] += jonswap.vel_vertical(
                        i, l, amplitude_pm[nwave])
                    jonswap_ac_horizontal[j] += jonswap.ac_horizontal(
                        i, l, amplitude_pm[nwave])
                    jonswap_ac_vertical[j] += jonswap.ac_vertical(
                        i, l, amplitude_pm[nwave])
                    nwave = nwave + 1
                j = j+1
            # plotagem dos gráficos
            # espectro de energia
            plt.clf()
            plt.title('Jonswap')

            if espectro == True:
                plt.plot(dw, pm, 'blueviolet',
                         label='Espectro de Onda')
                plt.xlabel('Frequência')
                plt.ylabel('Energia')
            else:
                pass

            # elevacao, velocidades e acelerações
            if elevacao == True:
                plt.plot(t1, jonswap_elevacao, 'orange',
                         label='Elevação de Superfíce')
                plt.xlabel('Tempo')
                plt.ylabel('Elevação')
            else:
                pass
            if vel_horizontal == True:
                plt.plot(t1, jonswap_vel_horizontal, 'b',
                         label='Velocidade Horizontal')
                plt.xlabel('Tempo')
                plt.ylabel('Velocidade')
            else:
                pass
            if vel_vertical == True:
                plt.plot(t1, jonswap_vel_vertical, 'g',
                         label='Velocidade Vertical')
                plt.xlabel('Tempo')
                plt.ylabel('Velocidade')
            else:
                pass
            if ac_horizontal == True:
                plt.plot(t1, jonswap_ac_horizontal, 'r',
                         label='Aceleração Horizontal')
                plt.xlabel('Tempo')
                plt.ylabel('Aceleração')
            else:
                pass
            if ac_vertical == True:
                plt.plot(t1, jonswap_ac_vertical, 'lime',
                         label='Aceleração Vertical')
                plt.xlabel('Tempo')
                plt.ylabel('Aceleração')
            else:
                pass
            plt.legend()
            plt.grid()
            plt.show()
