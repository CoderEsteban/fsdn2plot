#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fdsn2plot.py

Este script descarga datos sísmicos usando FDSN, los procesa con filtros básicos,
los convierte a aceleración y genera una imagen de las trazas.

✔️ Configuración auto-contenida.
✔️ Puede recibir por argumento el archivo TXT de estaciones.
✔️ Calcula altura de imagen dinámica.
✔️ Pensado para entornos reproducibles en GitHub.

Autor: Esteban López 
"""

# -------------------------------------------------------------------------
# 📌 IMPORTS
# -------------------------------------------------------------------------
import os
import sys
import warnings
import pytz
import numpy as np
import matplotlib.pyplot as plt
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy.io.mseed.headers import InternalMSEEDWarning
from matplotlib.dates import MinuteLocator, DateFormatter
from datetime import timedelta

warnings.filterwarnings("ignore", category=InternalMSEEDWarning)

# -------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# -------------------------------------------------------------------------

# Parámetros de conexión FDSN (ajusta a tu servidor SeisComP)
FDSN_SERVER_URL = "http://127.0.0.1:8080"   # URL del servidor FDSN
FDSN_USER = ""    # Usuario si aplica
FDSN_PASS = ""    # Clave si aplica

# Ventana de tiempo (segundos)
VENTANA_SEGUNDOS = 3600   # Última hora

# Archivos de entrada
#  1. archivo con estaciones por código: cada línea: "XX ABCD"
#  2. archivo con nombres de estaciones: cada línea: "ABCD Estación en la Provincia de Alajuela"
#  Estos se pasan por argumento o se editan aquí:
ARCHIVO_ESTACIONES_CODIGO = sys.argv[1] if len(sys.argv) > 1 else "lista_de_estaciones_por_codigo.txt"
ARCHIVO_ESTACIONES_NOMBRE = "lista_de_estaciones_por_nombre.txt"

# Tamaño base de imagen (ancho fijo)
IMG_WIDTH_INCHES = 9
IMG_DPI = 100
IMG_HEIGHT_PER_STATION = 0.5  # pulgadas por estación

# Nombre del archivo de salida (sin espacios)
OUTPUT_FILENAME = "fdsn2plot.png"

# -------------------------------------------------------------------------
# INICIALIZACIÓN
# -------------------------------------------------------------------------

# Conexión
client_fdsn = Client(base_url=FDSN_SERVER_URL)

ahora = UTCDateTime()
t_inicio = ahora - VENTANA_SEGUNDOS
t_fin = ahora

# Nombres de estaciones
estacion_nombres = {}
with open(ARCHIVO_ESTACIONES_NOMBRE, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split(None, 1)
            if len(parts) == 2:
                estacion_nombres[parts[0]] = parts[1]

# Estaciones por código
estaciones = []
with open(ARCHIVO_ESTACIONES_CODIGO, "r") as f:
    for line in f:
        partes = line.strip().split()
        if len(partes) >= 2:
            net, full_sta = partes[:2]
            loc = "*"
            cha = "H*"
            estaciones.append((net, full_sta, loc, cha))

if not estaciones:
    print("No hay estaciones definidas.")
    sys.exit(1)

# -------------------------------------------------------------------------
# FIGURA
# -------------------------------------------------------------------------

IMG_HEIGHT_INCHES = max(IMG_HEIGHT_PER_STATION * len(estaciones), 3)  # min. 3 pulgadas de alto

fig, ax = plt.subplots(
    nrows=len(estaciones),
    ncols=1,
    figsize=(IMG_WIDTH_INCHES, IMG_HEIGHT_INCHES),
    dpi=IMG_DPI,
    sharex=True
)

if len(estaciones) == 1:
    ax = [ax]

plt.subplots_adjust(left=0.15, right=0.75, hspace=0.4, top=0.95, bottom=0.05)
fig.patch.set_facecolor('white')

# -------------------------------------------------------------------------
# GRAFICAR
# -------------------------------------------------------------------------

for i, (net, full_sta, loc, cha) in enumerate(estaciones):
    splitted = full_sta.split('.')
    sta = splitted[1] if len(splitted) == 2 else full_sta
    nombre_estacion = estacion_nombres.get(sta, sta)

    try:
        st = client_fdsn.get_waveforms(
            network=net, station=sta, location=loc, channel=cha,
            starttime=t_inicio, endtime=t_fin, attach_response=True
        )

        st.remove_response(output="ACC", zero_mean=True, taper=True)
        st.filter("highpass", freq=0.1, corners=4, zerophase=True)
        st.filter("lowpass", freq=10.0, corners=4, zerophase=True)
        st.filter("bandstop", freqmin=49, freqmax=51, corners=2, zerophase=True)

        tr = st[0]
        data_cm_s2 = tr.data * 100.0

        tiempos = tr.times("matplotlib")
        mask = np.isfinite(data_cm_s2)
        lat_sec = max(0, ahora - tr.stats.endtime)
        lat_str = f"(latencia {lat_sec:.1f}s)" if lat_sec < 60 else f"(latencia {lat_sec / 60:.1f}m)"

        left_label = f"[{abs(data_cm_s2).max():.2f} cm/s²] {sta}"

        ax[i].plot(tiempos[mask], data_cm_s2[mask], 'b-')

    except Exception as e:
        left_label = f"[Sin datos] {sta}"
        nombre_estacion = f"{nombre_estacion} (sin datos)"
        lat_str = ""
        ax[i].plot([t_inicio.matplotlib_date, t_fin.matplotlib_date], [0, 0], 'r-')

    ax[i].text(-0.02, 1.0, left_label, transform=ax[i].transAxes, ha='right', va='top', color='black', fontsize=8)
    ax[i].text(1.02, 1.1, nombre_estacion, transform=ax[i].transAxes, ha='left', va='top', color='black', fontsize=8)
    if lat_str:
        ax[i].text(1.02, 0.35, lat_str, transform=ax[i].transAxes, ha='left', va='top', color='black', fontsize=5)

    ax[i].spines['top'].set_visible(False)
    ax[i].spines['bottom'].set_visible(False)
    ax[i].spines['left'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].yaxis.set_visible(False)
    ax[i].set_xlim(t_inicio.matplotlib_date, t_fin.matplotlib_date)

# -------------------------------------------------------------------------
# EJE X Y GUARDAR
# -------------------------------------------------------------------------

ax[-1].xaxis.set_major_locator(MinuteLocator(interval=10))
ax[-1].xaxis.set_minor_locator(MinuteLocator(interval=5))
ax[-1].xaxis.set_major_formatter(DateFormatter('%H:%M', tz=pytz.FixedOffset(-6*60)))
ax[-1].set_xlabel("Tiempo (Hora local)", fontsize=8)
fig.autofmt_xdate()

plt.savefig(OUTPUT_FILENAME, dpi=IMG_DPI, facecolor='white')
plt.show()

print(f"✅ Imagen generada: {OUTPUT_FILENAME}")
 
