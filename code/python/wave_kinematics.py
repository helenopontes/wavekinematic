from os import write
import PySimpleGUI as sg
import math
from math import pi, cos, sqrt, sin, cosh, sinh
import numpy as np
import pandas as pd
from scipy import integrate
import sympy as sp
import random
import matplotlib.pyplot as plt
from airy import Airy
from stokes import Stokes
from pierson_moskowitz import PM
from jonswap import Jonswap
import warnings
warnings.filterwarnings("ignore")

#Colocar data e hora
from datetime import datetime
##


# Criar janelas e layouts

sg.theme('LightGrey1')


def inicio():
    layout = [
        [sg.Text('Wave Kinematics é um programa com a finalidade de calcular as propriedades cinemáticas de ondas.')],
        [sg.Frame(layout=[[sg.Radio('Ondas Regulares', 'tipo', key='regular', default=True), sg.Radio(
            'Ondas Irregulares', 'tipo', key='irregular')]], title='Qual teoria de onda será utilizada?', title_color='black')],
        [sg.Button('Continuar'), sg.Button('Sobre')],
    ]
    return sg.Window(
        'Wave Kinematics - Inicio', layout=layout, finalize=True)

def sobre():
    layout = [
        [sg.Frame(layout=[[sg.Text('Autor: José Cícero de Assis Silva Junior\nVersão: 1.1\nCentro de Tecnologia-CTEC\nUniversidade Federal de Alagoas-UFAL\n2021')]], title='Sobre o aplicativo', title_color='black')],
        [sg.Button('Voltar')],
    ]
    return sg.Window(
        'About - Wave Kinematics', layout=layout, finalize=True)

def regular():
    layout = [
        [sg.Frame(layout=[[sg.Text('Altura(m)          '), sg.Input(size=(10, 0),
                                                                    justification='center', key='altura', tooltip='Altura da Onda')],
                          [sg.Text('Período(s)         '), sg.Input(size=(10, 0),
                                                                    justification='center', key='periodo', tooltip='Período da Onda')],
                          [sg.Text('Profundidade(m)'), sg.Input(size=(10, 0),
                                                                justification='center', key='profundidade', tooltip="Lâmina d'água")],
                          [sg.Text('Posição(m)       '), sg.Input(size=(10, 0),
                                                                  justification='center', key='posicao', tooltip="Posição horizontal(x)")],
                          [sg.Text('Instante(s)        '), sg.Input(size=(10, 0),
                                                                    justification='center', key='tempo', tooltip="Tempo Inicial")]], title='Insira os valores de:', title_color='black')],
        [sg.Frame(layout=[[sg.Radio('Airy', 'teoria', key='airy', default='True'),
                           sg.Radio('Stokes', 'teoria', key='stokes')]], title='Qual teoria de onda será usada?', title_color='black')],
        [sg.Frame(layout=[[sg.Radio('do Tempo', 'tipo', key='choice_tempo', default='True'), sg.Radio(
            'da Profundidade', 'tipo', key='choice_profundidade')]], title='Deseja analisar as propriedades ao longo:', title_color='black')],
        [sg.Frame(layout=[[sg.Checkbox('Elevação de Superfície', key='elevacao_regular')],
                          [sg.Checkbox('Velocidade Horizontal', key='vel_horizontal'), sg.Checkbox(
                              'Velocidade Vertical', key='vel_vertical')],
                          [sg.Checkbox('Aceleração Horizontal', key='ac_horizontal'), sg.Checkbox(
                              'Aceleração Vertical', key='ac_vertical')]], title='Assinale quais propriedades serão analisadas', title_color='black')],
        [sg.Button('Voltar'), sg.Button('Processar Dados'),
         sg.Button('Gerar Gráficos')],
        [sg.Text('Resultados:')],
        [sg.Output(size=(60, 11), key = '_output_')]
    ]
    return sg.Window(
        'Wave Kinematics - Ondas Regulares', layout=layout, finalize=True)


def irregular():
    layout = [
        [sg.Frame(layout=[[sg.Text('Altura (m)             '), sg.Input(size=(10, 0), justification='center', key='altura', tooltip='Altura Significativa')],
                          [sg.Text('Período (s)            '), sg.Input(
                              size=(10, 0), justification='center', key='periodo_pico', tooltip='Período de Pico')],
                          [sg.Text('Profundidade (m)   '), sg.Input(
                              size=(10, 0), justification='center', key='profundidade', tooltip="Lâmina d'água")],
                          [sg.Text(
                              'Qtd. de ondas       '), sg.Input(size=(10, 0), justification='center', key='qtdondas', tooltip='Recomendam-se 200 ondas')],
                          [sg.Text('Pos. horizontal (m)'), sg.Input(
                              size=(10, 0), justification='center', key='posicao_horizontal', tooltip='Posição horizontal (x)')],
                          [sg.Text('Pos. vertical (m)    '), sg.Input(
                              size=(10, 0), justification='center', key='posicao_vertical', tooltip='Posição vertical (z)')]], title='Insira os valores de:', title_color='black')],
        [sg.Frame(layout=[[sg.Radio('Pierson-Moskowitz', 'espectro', key='pierson_moskowitz', default=True),
                           sg.Radio('Jonswap', 'espectro', key='jonswap')]], title='Qual espectro de onda será utilizado?', title_color='black')],
        [sg.Frame(layout=[[sg.Checkbox('Espectro de Energia  ', key='espectro'), sg.Checkbox(
            'Elevacao de Superfície', key='elevacao')],
            [sg.Checkbox('Velocidade Horizontal', key='vel_horizontal'), sg.Checkbox(
                'Velocidade Vertical', key='vel_vertical')],
            [sg.Checkbox('Aceleração Horizontal', key='ac_horizontal'), sg.Checkbox(
                'Aceleração Vertical', key='ac_vertical')]], title='Assinale quais propriedades serão analisadas', title_color='black')],
        [sg.Button('Voltar'), sg.Button('Processar Dados'),
         sg.Button('Gerar Gráficos')],
        [sg.Text('Resultados:')],
        [sg.Output(size=(60, 11), key = '_output_')]
    ]
    return sg.Window(
        'Wave Kinematics - Ondas Irregulares', layout=layout, finalize=True)


# Criar janelas iniciais
janela1, janela2, janela3, janela4 = inicio(), None, None, None

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
    #Exibir o "sobre"
    elif window == janela1 and event == "Sobre":
        janela4 = sobre()
        janela1.hide()
    
    #ocultar sobre ou fechar o programa
    elif window == janela4 and event == sg.WIN_CLOSED:
        break
    elif window == janela4 and event == 'Voltar':
        janela4.hide()
        janela1.un_hide()
    
    # fechar segunda janela
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
        elevacao_regular = values['elevacao_regular']
        vel_horizontal = values['vel_horizontal']
        vel_vertical = values['vel_vertical']
        ac_horizontal = values['ac_horizontal']
        ac_vertical = values['ac_vertical']
        choice_tempo = values['choice_tempo']
        choice_profundidade = values['choice_profundidade']

        if airy == True and stokes == False:
            # Chamar a classe Airy para o cálculo das variáveis percentuais
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

            # Características da onda
            k = round(wave1.k, 2)
            c = round(wave1.c, 2)
            L = round(wave1.L, 2)
            
            #Limpando tela do output
            window.FindElement('_output_').Update('')
            
            print(
                f'Característica da onda: \nTeoria de Onda: Airy\nNúmero de Onda: {k} \nCeleridade: {c} \nComprimento de onda: {L}\n')

            # Calculo percentual das velocidades e acelerações
            PVh = round(100*(uhL2/uhL0), 2)
            PVv = round(100*(uvL2/uvL0), 2)
            PAv = round(100*(AvL2/AvL0), 2)
            PAh = round(100*(AhL2/AhL0), 2)

            # Expor o resultado das variáveis percentuais
            print(
                f'Variáveis percentuais: \nVelocidade Horizontal: {PVh}% \nVelocidade Vertical: {PVv}% \nAceleração Horizontal: {PAh}% \nAceleração Vertical: {PAv}%')

            # Criação de gráficos
            if choice_tempo == True and choice_profundidade == False:
                # Plotando ao longo do tempo
                # Criação do vetor de tempo e dos vetores vazios de velocidades e acelerações
                t1 = np.arange(0, 100, 0.1)
                elevacao_t = np.zeros(t1.size)
                vel_horizontal_t = np.zeros(t1.size)
                vel_vertical_t = np.zeros(t1.size)
                ac_horizontal_t = np.zeros(t1.size)
                ac_vertical_t = np.zeros(t1.size)

                #Wheeler Stretching
                z0 = 0

                # Substituição dos valores nos vetores
                j = 0
                for i in t1:
                    elevacao_t[j] = wave1.elevacao(i, x)
                    #Wheeler Stretching - DNV-RP-C205
                    z = (z0 - elevacao_t[j])/(1 + elevacao_t[j]/d)
                    vel_horizontal_t[j] = wave1.vel_horizontal(i, x, z)
                    vel_vertical_t[j] = wave1.vel_vertical(i, x, z)
                    ac_horizontal_t[j] = wave1.ac_horizontal(i, x, z)
                    ac_vertical_t[j] = wave1.ac_vertical(i, x, z)
                    j = j+1

                # Gravar dados em excel
                dados_excel1 = {'Tempo': t1, 'Elevação': elevacao_t, 'Velocidade Horizontal': vel_horizontal_t,
                                'Velocidade Vertical': vel_vertical_t, 'Aceleração Horizontal': ac_horizontal_t, 'Aceleração Vertical': ac_vertical_t}
                excel1 = pd.DataFrame(dados_excel1)
                #Colocando Data e Hora no nome do arquivo
                data_hora = datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
                excel_airy_tempo = "Regular_Airy_Tempo"
                excel_extensao = ".xlsx"
                excel1.to_excel(excel_airy_tempo + data_hora + excel_extensao)                

            else:
                # Plotando ao longo da profundidade
                # Criação do vetor de profundidade e dos vetores vazios de velocidades e acelerações
                z = int(d)
                z1 = np.arange(0, z, 1)
                vel_horizontal_z = np.zeros(z1.size)
                vel_vertical_z = np.zeros(z1.size)
                ac_horizontal_z = np.zeros(z1.size)
                ac_vertical_z = np.zeros(z1.size)

                # Substituição dos valores nos vetores
                for i in z1:
                    vel_horizontal_z[i] = wave1.vel_horizontal(t, x, -i)
                    vel_vertical_z[i] = wave1.vel_vertical(t, x, -i)
                    ac_horizontal_z[i] = wave1.ac_horizontal(t, x, -i)
                    ac_vertical_z[i] = wave1.ac_vertical(t, x, -i)

                # Gravar dados em excel
                dados_excel2 = {'Profundidade': z1, 'Velocidade Horizontal': vel_horizontal_z, 'Velocidade Vertical': vel_vertical_z,
                                'Aceleração Horizontal': ac_horizontal_z, 'Aceleração Vertical': ac_vertical_z}
                excel2 = pd.DataFrame(dados_excel2)
                
                #Colocando Data e Hora no nome do arquivo
                data_hora = datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
                excel_airy_profundidade = "Regular_Airy_Profundidade"
                excel_extensao = ".xlsx"
                excel2.to_excel(excel_airy_profundidade + data_hora + excel_extensao)

        elif airy == False and stokes == True:
            # Chamar a classe Airy para o cálculo das variáveis percentuais
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

            # Características da onda
            k = round(wave2.k, 2)
            c = round(wave2.c, 2)
            L = round(wave2.L, 2)
            
            #Limpando tela do output
            window.FindElement('_output_').Update('')            
            
            print(
                f'Característica da onda: \nTeoria de Onda: Stokes\nNúmero de Onda: {k} \nCeleridade: {c} \nComprimento de onda: {L}\n')

            # Calculo percentual das velocidades e acelerações
            PsVh = round(100*(uthL2/uthL0), 2)
            PsVv = round(100*(vthL2/vthL0), 2)
            PsAv = round(100*(AtuL2/AtuL0), 2)
            PsAh = round(100*(AthL2/AthL0), 2)

            # Expor o resultado das variáveis percentuais
            print(
                f'Variáveis percentuais: \nVelocidade Horizontal: {PsVh}% \nVelocidade Vertical: {PsVv}% \nAceleração Horizontal: {PsAh}% \nAceleração Vertical: {PsAv}% ')

            # Plotando o gráficos
            if choice_tempo == True and choice_profundidade == False:
                # Plotando ao longo do tempo
                # Criação do vetor de tempo e dos vetores vazios de velocidades e acelerações
                t1 = np.arange(0, 100, 0.1)
                elevacao_t = np.zeros(t1.size)
                vel_horizontal_t = np.zeros(t1.size)
                vel_vertical_t = np.zeros(t1.size)
                ac_horizontal_t = np.zeros(t1.size)
                ac_vertical_t = np.zeros(t1.size)

                #Wheeler Stretching
                z0 = 0

                # Substituição dos valores nos vetores
                j = 0
                for i in t1:
                    elevacao_t[j] = wave2.elevacao(i, x)
                    #Wheeler Stretching - DNV-RP-C205
                    z = (z0 - elevacao_t[j])/(1 + elevacao_t[j]/d)                    
                    vel_horizontal_t[j] = wave2.vel_horizontal(i, x, z)
                    vel_vertical_t[j] = wave2.vel_vertical(i, x, z)
                    ac_horizontal_t[j] = wave2.ac_horizontal(i, x, z)
                    ac_vertical_t[j] = wave2.ac_vertical(i, x, z)
                    j = j+1

                # Gravar arquivos em excel
                dados_excel3 = {'Tempo': t1, 'Elevação': elevacao_t, 'Velocidade Horizontal': vel_horizontal_t,
                                'Velocidade Vertical': vel_vertical_t, 'Aceleração Horizontal': ac_horizontal_t, 'Aceleração Vertical': ac_vertical_t}
                excel3 = pd.DataFrame(dados_excel3)
                
                #Colocando Data e Hora no nome do arquivo
                data_hora = datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
                excel_stokes_tempo = "Regular_Stokes_Tempo"
                excel_extensao = ".xlsx"
                excel3.to_excel(excel_stokes_tempo + data_hora + excel_extensao)
                
                #excel3.to_excel('stokes_tempo.xlsx')

            else:
                # Plotando ao longo da profundidade
                # Criação do vetor de profundidade e dos vetores vazios de velocidades e acelerações
                z = int(d)
                z1 = np.arange(0, z, 1)
                vel_horizontal_z = np.zeros(z1.size)
                vel_vertical_z = np.zeros(z1.size)
                ac_horizontal_z = np.zeros(z1.size)
                ac_vertical_z = np.zeros(z1.size)

                # Substituição dos valores nos vetores
                for i in z1:
                    vel_horizontal_z[i] = wave2.vel_horizontal(t, x, -i)
                    vel_vertical_z[i] = wave2.vel_vertical(t, x, -i)
                    ac_horizontal_z[i] = wave2.ac_horizontal(t, x, -i)
                    ac_vertical_z[i] = wave2.ac_vertical(t, x, -i)

                # Gravar dados em excel
                dados_excel4 = {'Profundidade': z1, 'Velocidade Horizontal': vel_horizontal_z, 'Velocidade Vertical': vel_vertical_z,
                                'Aceleração Horizontal': ac_horizontal_z, 'Aceleração Vertical': ac_vertical_z}
                excel4 = pd.DataFrame(dados_excel4)
                
                #Colocando Data e Hora no nome do arquivo
                data_hora = datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
                excel_stokes_profundidade = "Regular_Stokes_Profundidade"
                excel_extensao = ".xlsx"
                excel4.to_excel(excel_stokes_profundidade + data_hora + excel_extensao)

    elif window == janela2 and event == 'Gerar Gráficos':
        button, values = janela2.Read()
        airy = values['airy']
        stokes = values['stokes']
        elevacao_regular = values['elevacao_regular']
        vel_horizontal = values['vel_horizontal']
        vel_vertical = values['vel_vertical']
        ac_horizontal = values['ac_horizontal']
        ac_vertical = values['ac_vertical']
        choice_tempo = values['choice_tempo']
        choice_profundidade = values['choice_profundidade']

        if airy == True and stokes == False:
            if choice_tempo == True and choice_profundidade == False:
                plt.clf()

                #Elevação de Superfície em uma Janela 
                if elevacao_regular == True:
                    plt.title('Teoria de Airy')
                    plt.plot(t1, elevacao_t, 'r',
                             label='Elevação de Superfície')
                    plt.xlabel('Tempo (s)')
                    plt.ylabel('Elevação (m)')
                    plt.legend()
                else:
                    pass
                plt.grid()

                #Velocidades em uma Janela 
                if vel_horizontal == True:
                    figure2 = plt.figure('Figure 2.png')
                    plt.title('Teoria de Airy')
                    plt.plot(t1, vel_horizontal_t, 'b',
                             label='Velocidade Horizontal')
                    plt.xlabel('Tempo (s)')
                    plt.ylabel('Velocidade (m/s)')
                    plt.legend()
                else:
                    pass
                if vel_vertical == True:
                    figure2 = plt.figure('Figure 2.png')
                    plt.title('Teoria de Airy')
                    plt.plot(t1, vel_vertical_t, 'g',
                             label='Velocidade Vertical')
                    plt.xlabel('Tempo (s)')
                    plt.ylabel('Velocidade (m/s)')
                    plt.legend()
                else:
                    pass
                plt.grid()

                #Acelerações em uma Janela 
                if ac_horizontal == True:
                    figure3 = plt.figure('Figure 3.png')
                    plt.title('Teoria de Airy')
                    plt.plot(t1, ac_horizontal_t, 'b',
                             label='Aceleração Horizontal')
                    plt.xlabel('Tempo (s)')
                    plt.ylabel('Aceleração (m/s²)')
                    plt.legend()
                else:
                    pass
                if ac_vertical == True:
                    figure3 = plt.figure('Figure 3.png')
                    plt.title('Teoria de Airy')
                    plt.plot(t1, ac_vertical_t, 'g',
                             label='Aceleração Vertical')
                    plt.xlabel('Tempo (s)')
                    plt.ylabel('Aceleração (m/s²)')
                    plt.legend()    
                else:
                    pass
                plt.grid()
                plt.show()
            
            elif choice_tempo == False and choice_profundidade == True:
                plt.clf()
                if elevacao_regular == True:
                    pass
                else:
                    pass

                #Velocidades em uma Janela 
                if vel_horizontal == True:                    
                    plt.title('Teoria de Airy')
                    plt.plot(vel_horizontal_z, -z1, 'b',
                             label='Velocidade Horizontal')
                    plt.xlabel('Velocidade (m/s)')
                    plt.ylabel('Profundidade (m)')
                    plt.legend()
                else:
                    pass

                if vel_vertical == True:                    
                    plt.title('Teoria de Airy')
                    plt.plot(vel_vertical_z, -z1, 'g',
                             label='Velocidade Vertical')
                    plt.xlabel('Velocidade (m/s)')
                    plt.ylabel('Profundidade (m)')
                    plt.legend()
                else:
                    pass
                plt.grid()

                #Acelerações em uma Janela 
                if ac_horizontal == True:                    
                    plt.title('Teoria de Airy')
                    figure2 = plt.figure('Figure 2.png')
                    plt.plot(ac_horizontal_z, -z1, 'b',
                             label='Aceleração Horizontal')
                    plt.xlabel('Aceleração (m/s²)')
                    plt.ylabel('Profundidade (m)')
                    plt.legend()
                else:
                    pass
                if ac_vertical == True:
                    plt.title('Teoria de Airy')
                    figure2 = plt.figure('Figure 2.png')
                    plt.plot(ac_vertical_z, -z1, 'g',
                             label='Aceleração Vertical')
                    plt.xlabel('Aceleração (m/s²)')
                    plt.ylabel('Profundidade (m)')
                    plt.legend()
                else:
                    pass
                plt.grid()
                plt.show()

        elif airy == False and stokes == True:
            if choice_tempo == True and choice_profundidade == False:
                plt.clf()

                #Elevação em uma  Janela
                if elevacao_regular == True:
                    plt.title('Teoria de Stokes')
                    plt.plot(t1, elevacao_t, 'r',
                             label='Elevação de Superfície')
                    plt.xlabel('Tempo (s)')
                    plt.ylabel('Elevação (m)')
                    plt.legend()
                else:
                    pass
                plt.grid()

                #Velocidades em uma Janela
                if vel_horizontal == True:
                    plt.title('Teoria de Stokes')
                    figure2 = plt.figure('Figure 2.png')
                    plt.plot(t1, vel_horizontal_t, 'b',
                             label='Velocidade Horizontal')
                    plt.xlabel('Tempo (s)')
                    plt.ylabel('Velocidade (m/s)')
                    plt.legend()
                else:
                    pass
                if vel_vertical == True:
                    plt.title('Teoria de Stokes')
                    figure2 = plt.figure('Figure 2.png')
                    plt.plot(t1, vel_vertical_t, 'g',
                             label='Velocidade Vertical')
                    plt.xlabel('Tempo (s)')
                    plt.ylabel('Velocidade (m/s)')
                    plt.legend()
                else:
                    pass
                plt.grid()

                #Acelerações em uma Janela
                if ac_horizontal == True:
                    plt.title('Teoria de Stokes')
                    figure3 = plt.figure('Figure 3.png')
                    plt.plot(t1, ac_horizontal_t, 'b',
                             label='Aceleração Horizontal')
                    plt.xlabel('Tempo (s)')
                    plt.ylabel('Aceleração (m/s²)')
                    plt.legend()
                else:
                    pass
                if ac_vertical == True:
                    plt.title('Teoria de Stokes')
                    figure3 = plt.figure('Figure 3.png')
                    plt.plot(t1, ac_vertical_t, 'g',
                             label='Aceleração Vertical')
                    plt.xlabel('Tempo (s)')
                    plt.ylabel('Aceleração (m/s²)')
                    plt.legend()
                else:
                    pass
                plt.grid()
                plt.show()

            elif choice_tempo == False and choice_profundidade == True:
                plt.clf()
                if elevacao_regular == True:
                    pass
                else:
                    pass

                #Velocidades em uma Janela
                if vel_horizontal == True:
                    plt.title('Teoria de Stokes')
                    plt.plot(vel_horizontal_z, -z1, 'b',
                             label='Velocidade Horizontal')
                    plt.xlabel('Velocidade (m/s)')
                    plt.ylabel('Profundidade (m)')
                    plt.legend()
                else:
                    pass
                if vel_vertical == True:
                    plt.title('Teoria de Stokes')
                    plt.plot(vel_vertical_z, -z1, 'g',
                             label='Velocidade Vertical')
                    plt.xlabel('Velocidade (m/s)')
                    plt.ylabel('Profundidade (m)')
                    plt.legend()
                else:
                    pass
                plt.grid()

                #Acelerações em uma Janela
                if ac_horizontal == True:
                    plt.title('Teoria de Stokes')
                    figure2 = plt.figure('Figure 2.png')
                    plt.plot(ac_horizontal_z, -z1, 'b',
                             label='Aceleração Horizontal')
                    plt.xlabel('Aceleração (m/s²)')
                    plt.ylabel('Profundidade (m)')
                    plt.legend()
                else:
                    pass
                if ac_vertical == True:
                    plt.title('Teoria de Stokes')
                    figure2 = plt.figure('Figure 2.png')
                    plt.plot(ac_vertical_z, -z1, 'g',
                             label='Aceleração Vertical')
                    plt.xlabel('Aceleração (m/s²)')
                    plt.ylabel('Profundidade (m)')
                    plt.legend()
                else:
                    pass
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
        nOndas = int(values['qtdondas'])
        x = float(values['posicao_horizontal'])
        z = float(values['posicao_vertical'])
        choice_pm = values['pierson_moskowitz']
        choice_jonswap = values['jonswap']
        espectro = values['espectro']
        elevacao = values['elevacao']
        vel_horizontal = values['vel_horizontal']
        vel_vertical = values['vel_vertical']
        ac_horizontal = values['ac_horizontal']
        ac_vertical = values['ac_vertical']
        
        #Capturando data e hora
        data_hora = datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')

        if choice_pm == True and choice_jonswap == False:
            pierson_moskowitz = PM(d, H, Tp, nOndas, x, z, data_hora)
            # Momentos espectrais, Período de Cruzamento de Zeros e Desvio Padrão
            sp.init_printing(pretty_print=True)
            oo = math.inf
            m0, err1 = integrate.quad(
                pierson_moskowitz.momento_espectral, 0.01, oo)
            m2, err2 = integrate.quad(
                pierson_moskowitz.momento_espectral_2, 0.01, oo)
            m4, err3 = integrate.quad(
                pierson_moskowitz.momento_espectral_4, 0.01, oo)
            sig0 = sqrt(m0)
            Tz = 2*pi*sqrt(m0/m2)
            banda = sqrt(1-((m2**2)/(m0*m4)))
            
            #Limpando tela do output
            window.FindElement('_output_').Update('')            
            
            print(
                f'Características espectrais: \nEspectro: Pierson-Moskowitz\nMomento de ordem 0: {m0} \nMomento de ordem 2: {m2} \nMomento de ordem 4: {m4} \nPeríodo médio de cruzamento zero: {Tz} \nDesvio padrão: {sig0}')
            if banda < 0.6:
                print('Espectro de banda estreita')
            else:
                print('Espectro de banda larga')

            # Gráficos
            t1 = np.arange(0, 100, 0.1)
            pm_elevacao = np.zeros(t1.size)
            pm_vel_horizontal = np.zeros(t1.size)
            pm_vel_vertical = np.zeros(t1.size)
            pm_ac_horizontal = np.zeros(t1.size)
            pm_ac_vertical = np.zeros(t1.size)
            
            #Wheeler Stretching
            z0 = 0
            
            j = 0
            for l in t1:
                nwave = 0
                for i in pierson_moskowitz.w:
                    pm_elevacao[j] += pierson_moskowitz.elevacao(
                        i, l, pierson_moskowitz.A[nwave], nwave)

                    #Wheeler Stretching - DNV-RP-C205
                    z = (z0 - pm_elevacao[j])/(1 + pm_elevacao[j]/d)                        
                        
                    pm_vel_horizontal[j] += pierson_moskowitz.vel_horizontal(
                        i, l, pierson_moskowitz.A[nwave], nwave, z)
                    pm_vel_vertical[j] += pierson_moskowitz.vel_vertical(
                        i, l, pierson_moskowitz.A[nwave], nwave, z)
                    pm_ac_horizontal[j] += pierson_moskowitz.ac_horizontal(
                        i, l, pierson_moskowitz.A[nwave], nwave, z)
                    pm_ac_vertical[j] += pierson_moskowitz.ac_vertical(
                        i, l, pierson_moskowitz.A[nwave], nwave, z)
                    nwave = nwave + 1
                j = j + 1
            # Gravar arquivos em excel
            dados_excel5 = {'Tempo': t1, 'Elevação': pm_elevacao, 'Velocidade Horizontal': pm_vel_horizontal,
                            'Velocidade Vertical': pm_vel_vertical, 'Aceleração Horizontal': pm_ac_horizontal, 'Aceleração Vertical': pm_ac_vertical}
            excel5 = pd.DataFrame(dados_excel5)
            
            #Colocando Data e Hora no nome do arquivo
            #data_hora = datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
            excel_pm_cinematica = "Irregular_PM_Cinematica"
            excel_extensao = ".xlsx"
            excel5.to_excel(excel_pm_cinematica + data_hora + excel_extensao)            
            
        else:
            jonswap = Jonswap(d, H, Tp, nOndas, x, z, data_hora)
            # Momentos espectrais, Período de Cruzamento de Zeros e Desvio Padrão
            sp.init_printing(pretty_print=True)
            oo = math.inf
            m0, err1 = integrate.quad(
                jonswap.momento_espectral, 0.01, oo)
            m2, err2 = integrate.quad(
                jonswap.momento_espectral_2, 0.01, oo)
            m4, err3 = integrate.quad(
                jonswap.momento_espectral_4, 0.01, oo)
            sig0 = sqrt(m0)
            Tz = 2*pi*sqrt(m0/m2)
            banda = sqrt(1-((m2**2)/(m0*m4)))
            
            #Limpando tela do output
            window.FindElement('_output_').Update('')             
            
            print(
                f'Características espectrais: \nEspectro: JONSWAP\nMomento de ordem 0: {m0} \nMomento de ordem 2: {m2} \nMomento de ordem 4: {m4} \nPeríodo médio de cruzamento zero: {Tz} \nDesvio padrão: {sig0}')
            if banda < 0.6:
                print('Espectro de banda estreita')
            else:
                print('Espectro de banda larga')

            # Gráficos
            t1 = np.arange(0, 100, 0.1)
            jonswap_elevacao = np.zeros(t1.size)
            jonswap_vel_horizontal = np.zeros(t1.size)
            jonswap_vel_vertical = np.zeros(t1.size)
            jonswap_ac_horizontal = np.zeros(t1.size)
            jonswap_ac_vertical = np.zeros(t1.size)
            
            #Wheeler Stretching
            z0 = 0            
            
            j = 0
            for l in t1:
                nwave = 0
                for i in jonswap.w:
                    jonswap_elevacao[j] += jonswap.elevacao(
                        i, l, jonswap.A[nwave], nwave)

                    #Wheeler Stretching - DNV-RP-C205
                    z = (z0 - jonswap_elevacao[j])/(1 + jonswap_elevacao[j]/d)                        
                                             
                    jonswap_vel_horizontal[j] += jonswap.vel_horizontal(
                        i, l, jonswap.A[nwave], nwave, z)
                    jonswap_vel_vertical[j] += jonswap.vel_vertical(
                        i, l, jonswap.A[nwave], nwave, z)
                    jonswap_ac_horizontal[j] += jonswap.ac_horizontal(
                        i, l, jonswap.A[nwave], nwave, z)
                    jonswap_ac_vertical[j] += jonswap.ac_vertical(
                        i, l, jonswap.A[nwave], nwave, z)
                    nwave = nwave + 1
                j = j+1
            # Gravar arquivos em excel
            dados_excel6 = {'Tempo': t1, 'Elevação': jonswap_elevacao, 'Velocidade Horizontal': jonswap_vel_horizontal,
                            'Velocidade Vertical': jonswap_vel_vertical, 'Aceleração Horizontal': jonswap_ac_horizontal, 'Aceleração Vertical': jonswap_ac_vertical}
            excel6 = pd.DataFrame(dados_excel6)
            
            #Colocando Data e Hora no nome do arquivo
            #data_hora = datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
            excel_jonswap_cinematica = "Irregular_Jp_Cinematica"
            excel_extensao = ".xlsx"
            excel6.to_excel(excel_jonswap_cinematica + data_hora + excel_extensao)            

    elif window == janela3 and event == 'Gerar Gráficos':
        button, values = janela3.Read()
        choice_pm = values['pierson_moskowitz']
        choice_jonswap = values['jonswap']
        espectro = values['espectro']
        elevacao = values['elevacao']
        vel_horizontal = values['vel_horizontal']
        vel_vertical = values['vel_vertical']
        ac_horizontal = values['ac_horizontal']
        ac_vertical = values['ac_vertical']

        if choice_pm == True and choice_jonswap == False:
            plt.clf()
            #Separando em janelas
            if espectro == True:
                plt.title('Espectro de Pierson-Moskowitz')
                plt.plot(pierson_moskowitz.w, pierson_moskowitz.pm, 'r',
                         label='Espectro de Onda')
                plt.xlabel('Frequência (Hz)')
                plt.ylabel('Energia (m²/Hz)')
                plt.legend()
            else:
                pass
            plt.grid()

            #Janela da elevação
            if elevacao == True:
                plt.title('Espectro de Pierson-Moskowitz')
                figure2 = plt.figure('Figure 2.png')
                plt.plot(t1, pm_elevacao, 'r',
                         label='Elevação de Superfíce')
                plt.xlabel('Tempo (s)')
                plt.ylabel('Elevação (m)')
                plt.legend()
            else:
                pass
            plt.grid()

            #Janela das velocidades
            if vel_horizontal == True:
                plt.title('Espectro de Pierson-Moskowitz')
                figure3 = plt.figure('Figure 3.png')
                plt.plot(t1, pm_vel_horizontal, 'b',
                         label='Velocidade Horizontal')
                plt.xlabel('Tempo (s)')
                plt.ylabel('Velocidade (m/s)')
                plt.legend()
            else:
                pass
            if vel_vertical == True:
                plt.title('Espectro de Pierson-Moskowitz')
                figure3 = plt.figure('Figure 3.png')
                plt.plot(t1, pm_vel_vertical, 'g', label='Velocidade Vertical')
                plt.xlabel('Tempo (s)')
                plt.ylabel('Velocidade (m/s)')
                plt.legend()
            else:
                pass
            plt.grid()
            #Janela das acelerações
            if ac_horizontal == True:
                plt.title('Espectro de Pierson-Moskowitz')
                figure4 = plt.figure('Figure 4.png')
                plt.plot(t1, pm_ac_horizontal, 'b',
                         label='Aceleração Horizontal')
                plt.xlabel('Tempo (s)')
                plt.ylabel('Aceleração (m/s²)')
                plt.legend()
            else:
                pass
            if ac_vertical == True:
                plt.title('Espectro de Pierson-Moskowitz')
                figure4 = plt.figure('Figure 4.png')
                plt.plot(t1, pm_ac_vertical, 'g',
                         label='Aceleração Vertical')
                plt.xlabel('Tempo (s)')
                plt.ylabel('Aceleração (m/s²)')
                plt.legend()
            else:
                pass
            plt.grid()
            plt.show()

        elif choice_pm == False and choice_jonswap == True:
            plt.clf()
            #Separando os gráficos em janelas
            if espectro == True:
                plt.title('Espectro de JONSWAP')
                plt.plot(jonswap.w, jonswap.jp, 'r',
                         label='Espectro de Onda')
                plt.xlabel('Frequência (Hz)')
                plt.ylabel('Energia (m²/Hz)')
                plt.legend()
            else:
                pass
            plt.grid()

            #Janela da elevação
            if elevacao == True:
                plt.title('Espectro de JONSWAP')
                figure2 = plt.figure('Figure 2.png')
                plt.plot(t1, jonswap_elevacao, 'r',
                         label='Elevação de Superfíce')
                plt.xlabel('Tempo (s)')
                plt.ylabel('Elevação (m)')
                plt.legend()
            else:
                pass
            plt.grid()

            #Janela das velocidades
            if vel_horizontal == True:
                plt.title('Espectro de JONSWAP')
                figure3 = plt.figure('Figure 3.png')
                plt.plot(t1, jonswap_vel_horizontal, 'b',
                         label='Velocidade Horizontal')
                plt.xlabel('Tempo (s)')
                plt.ylabel('Velocidade (m/s)')
                plt.legend()
            else:
                pass
            if vel_vertical == True:
                plt.title('Espectro de JONSWAP')
                figure3 = plt.figure('Figure 3.png')
                plt.plot(t1, jonswap_vel_vertical, 'g',
                         label='Velocidade Vertical')
                plt.xlabel('Tempo (s)')
                plt.ylabel('Velocidade (m/s)')
                plt.legend()
            else:
                pass
            plt.grid()

            #Janela das acelerações
            if ac_horizontal == True:
                plt.title('Espectro de JONSWAP')
                figure4 = plt.figure('Figure 4.png')
                plt.plot(t1, jonswap_ac_horizontal, 'b',
                         label='Aceleração Horizontal')
                plt.xlabel('Tempo (s)')
                plt.ylabel('Aceleração (m/s²)')
                plt.legend()
            else:
                pass
            if ac_vertical == True:
                plt.title('Espectro de JONSWAP')
                figure4 = plt.figure('Figure 4.png')
                plt.plot(t1, jonswap_ac_vertical, 'g',
                         label='Aceleração Vertical')
                plt.xlabel('Tempo (s)')
                plt.ylabel('Aceleração (m/s²)')
                plt.legend()
            else:
                pass
            plt.grid()
            plt.show()
