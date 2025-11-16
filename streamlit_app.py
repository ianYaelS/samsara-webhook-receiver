# --------------------------------------------------------------------------
# app.py
#
# Aplicación Principal del Dashboard "Reefer-Tech" con Streamlit.
#
# v66 (Ajustes de UI Fina)
# 1. (UI) Log de alertas: Añadido 'max_chars=40' a 'Tipo de alerta'
#    para evitar que la tabla se ensanche con texto largo.
# 2. (UI) Revertido el st.expander a st.container para el log de alertas.
# 3. (UI) Log de alertas: use_container_width=False para tabla ajustada.
# 4. (UI) Log de alertas: Reordenadas columnas (Tipo, Incidente, Hora, Ref).
# --------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import folium
from folium.features import DivIcon
from streamlit_folium import st_folium
import pytz
from datetime import datetime, timedelta

# Importar nuestras utilidades de backend (API, IA)
import utils

# Importar el componente de auto-refresco
from streamlit_autorefresh import st_autorefresh

# --- Imports añadidos para Webhooks y DB ---
import os
import databases
import sqlalchemy
from sqlalchemy import Column, Integer, String, DateTime, JSON, MetaData
from dotenv import load_dotenv
import asyncio
# --- Fin de imports modificados ---

# --- 1. CONFIGURACIÓN INICIAL DE LA PÁGINA Y DEL ESTADO ---

st.set_page_config(
    page_title="Samsara Reefer Dashboard",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MEXICO_TZ = pytz.timezone("America/Mexico_City") 

# v59: Intervalo de refresco reducido a 20 segundos para mayor fluidez.
REFRESH_INTERVAL_SEC = 20

# --- (MODIFICADO) Conexión a Base de Datos (para Webhooks) ---
load_dotenv() # Asegurarse que .env se carga
DATABASE_URL = os.getenv("DATABASE_URL")

# FIX: Asegurar que la URL sea compatible con SQLAlchemy, ya sea en local o en Render.
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# ROBUSTEZ LOCAL: Si DATABASE_URL no existe (ej. corriendo localmente), 
# se usa SQLite persistente ('sqlite:///alerts.db'). 
if not DATABASE_URL:
    st.warning("ADVERTENCIA: DATABASE_URL no está configurada (modo local). Usando SQLite local (alerts.db) para el log de alertas.")
    DATABASE_URL = "sqlite:///alerts.db" 


# --- FIX CRÍTICO: Reemplazar @st.on_event con @st.cache_resource ---

# --- (MODIFICADO v65) ---
# Definición de la tabla 'alerts' (globalmente para ambas funciones)
alerts = sqlalchemy.Table(
    "alerts",
    MetaData(), # Usamos MetaData temporal aquí
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("event_id", String, unique=True, index=True),
    Column("timestamp", DateTime(timezone=True)),
    Column("vehicle_id", String),
    Column("vehicle_name", String), # <--- Esta columna ahora es "Referencia"
    Column("alert_type", String),
    Column("message", JSON),
    Column("incident_url", String, nullable=True), # <--- NUEVA COLUMNA
    Column("raw_json", JSON),
)

@st.cache_resource
def get_database_connection(db_url):
    """
    Inicializa la conexión a la base de datos y la retorna como recurso cacheado.
    """
    db = databases.Database(db_url)
    print(f"Base de datos inicializada con cache_resource: {db_url}")
    return db

database = get_database_connection(DATABASE_URL)
metadata_db = alerts.metadata # Usamos el metadata asociado a la tabla alerts

# --- INICIALIZACIÓN DE SESSION_STATE ---
try:
    if 'api_client' not in st.session_state:
        # Nota: SAMSARA_API_KEY debe estar en un archivo .env si se corre localmente
        if not utils.SAMSARA_API_KEY:
            # st.error ya se maneja en el archivo utils.py
            pass
        st.session_state.api_client = utils.SamsaraAPIClient(api_key=utils.SAMSARA_API_KEY)
    if 'ai_model' not in st.session_state:
        st.session_state.ai_model = utils.AIModels()

except ValueError as e:
    st.error(f"Error de Configuración: {e}. Revisa tu archivo .env.")
    st.stop()
except Exception as e:
    st.error(f"Error fatal al inicializar: {e}")
    st.stop

# Inicializar estado de la UI
if 'selected_vehicle_name' not in st.session_state:
    st.session_state.selected_vehicle_name = None
if 'selected_vehicle_obj' not in st.session_state:
    st.session_state.selected_vehicle_obj = None
if 'sensor_config' not in st.session_state:
    st.session_state.sensor_config = None
if 'last_webhook_timestamp' not in st.session_state:
    st.session_state.last_webhook_timestamp = None

# Nuevo estado para notificación de alertas
if 'last_alert_id' not in st.session_state:
    st.session_state.last_alert_id = None
    
# --- CSS ---
st.markdown("""
<style>
    /* Ajustes del contenedor principal (si es necesario) */
    .block-container { 
        padding-top: 2rem; padding-bottom: 2rem; 
        padding-left: 3rem; padding-right: 3rem;
    }

    /* Asegurar que todos los contenedores de KPI en la primera fila sean de la misma altura */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]:nth-child(1) {
        height: 100%;
    }
    
    /* Estilos de KPIs (st.metric) - Usado por Temp, Hum, Bat */
    .stMetric {
        border-bottom: 1px solid #262730; 
        padding-bottom: 0.5rem;
    }
    
    /* FIX: Para el mapa y alertas, forzar a que la fila ocupe el espacio */
    div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] {
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)


# --- 2. TÍTULO Y AUTO-REFRESCO ---

st.title("❄️ Samsara Reefer-Tech")
# v59: Caption actualizado para reflejar el nuevo intervalo de refresco
st.caption(f"Monitoreo en tiempo real de temperatura, puertas y GPS. (Refresca cada {REFRESH_INTERVAL_SEC}s)")

# Auto-refresco global para datos de API
st_autorefresh(interval=REFRESH_INTERVAL_SEC * 1000, limit=None, key="data_refresher")


# --- 3. FUNCIONES DE CARGA DE DATOS (CACHEADAS) ---

# Cachear lista de vehículos por 10 minutos
@st.cache_data(ttl=600)
def load_vehicle_list(_api_client):
    if not _api_client: return [], {}
    vehicles = _api_client.get_vehicles()
    vehicle_map_obj = {v['name']: v for v in vehicles}
    vehicle_names = [v['name'] for v in vehicles]
    return vehicle_names, vehicle_map_obj

# v59: TTL (cache) se ata automáticamente a REFRESH_INTERVAL_SEC (20s)
@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
def fetch_live_kpis(_api_client, sensor_config, vehicle_id):
    """(v43) Esta función ahora solo obtiene el Snapshot
    para el Mapa (GPS) y el Check Engine Light (Faults)."""
    if not _api_client or not vehicle_id: return None
    stats_data = _api_client.get_live_stats(vehicle_id)
    return stats_data

# v59: TTL (cache) se ata automáticamente a REFRESH_INTERVAL_SEC (20s)
@st.cache_data(ttl=REFRESH_INTERVAL_SEC) 
def fetch_live_sensor_history(_api_client, sensor_config, window_minutes, step_seconds):
    """(v37) Esta función es solo para SENSORES (Temp, Hum, Puerta)."""
    if not _api_client or not sensor_config:
        return [], {}
    return _api_client.get_live_sensor_history(
        sensor_config, 
        window_minutes, 
        step_seconds
    )

# v59: TTL (cache) se ata automáticamente a REFRESH_INTERVAL_SEC (20s)
@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
def fetch_vehicle_stats_history(_api_client, vehicle_id, window_minutes):
    """(v37) Esta función es para Batería, GPS y Fallas."""
    if not _api_client or not vehicle_id:
        return None
    return _api_client.get_vehicle_stats_history(
        vehicle_id,
        window_minutes
    )

@st.cache_data
def process_sensor_history_data(results, column_map, step_seconds=30):
    """(v37) Procesa /sensors/history."""
    if not results or not column_map:
        return pd.DataFrame() 

    data_rows = []
    has_temp = "temperature" in column_map.values()
    has_hum = "humidity" in column_map.values()
    has_door = "doorClosed" in column_map.values()

    for point in results:
        row = {'timestamp': pd.to_datetime(point['timeMs'], unit='ms')}
        temp_val = None
        for i, value in enumerate(point['series']):
            col_name = column_map.get(i)
            if col_name and value is not None:
                if col_name == 'temperature':
                    temp_val = float(value) / 1000.0
                elif col_name == 'humidity':
                    row['humidity'] = float(value)
                elif col_name == 'doorClosed':
                    # v55: Asegurar que el valor sea numérico para el gráfico
                    row['doorClosed'] = 1.0 if value else 0.0
        if temp_val is not None:
            row['temperature'] = temp_val
        data_rows.append(row)
        
    if not data_rows: 
        return pd.DataFrame()
    
    df = pd.DataFrame(data_rows).set_index('timestamp').sort_index()
    if has_temp and 'temperature' not in df.columns: df['temperature'] = np.nan
    if has_hum and 'humidity' not in df.columns: df['humidity'] = np.nan
    if has_door and 'doorClosed' not in df.columns: df['doorClosed'] = np.nan
    if df.empty:
        return pd.DataFrame()

    df = df.tz_localize(pytz.utc).tz_convert(MEXICO_TZ)
    
    continuous_cols = [col for col in ['temperature', 'humidity'] if col in df.columns]
    df_continuous = df[continuous_cols].interpolate(method='time')
    
    # v55: Tratar la puerta como "step" (ffill)
    discrete_cols = [col for col in ['doorClosed'] if col in df.columns]
    df_discrete = df[discrete_cols].ffill().bfill() 
    
    df = pd.concat([df_continuous, df_discrete], axis=1)
    
    return df.ffill().bfill()

@st.cache_data
def process_vehicle_stats_history(stats_history_data):
    """(v37) Procesa la respuesta de /stats/history (Batería y Fallas)."""
    df_battery = pd.DataFrame()
    all_faults_list = []

    if not stats_history_data or 'data' not in stats_history_data or not stats_history_data['data']:
        return df_battery, all_faults_list

    vehicle_data = stats_history_data['data'][0]

    if 'batteryMilliVolts' in vehicle_data:
        bat_history = vehicle_data['batteryMilliVolts']
        if bat_history:
            df_battery_rows = [
                {'timestamp': pd.to_datetime(d['time']), 'value': float(d['value']) / 1000.0}
                for d in bat_history if d.get('value') is not None
            ]
            if df_battery_rows:
                df_battery = pd.DataFrame(df_battery_rows).set_index('timestamp').sort_index()
                
                if not df_battery.index.tz:
                     df_battery = df_battery.tz_localize(pytz.utc).tz_convert(MEXICO_TZ)
                else:
                     df_battery = df_battery.tz_convert(MEXICO_TZ)
                
                df_battery = df_battery.rename(columns={'value': 'value'})

    if 'faultCodes' in vehicle_data:
        all_faults_list = vehicle_data['faultCodes']

    return df_battery, all_faults_list

# --- (v42) FUNCIONES DE CARGA DE ALERTAS API (Para KPI de Puerta) ---
@st.cache_data(ttl=600)
def load_alert_configurations(_api_client):
    if not _api_client: return None
    return _api_client.get_alert_configurations()

# v59: TTL (cache) se ata automáticamente a REFRESH_INTERVAL_SEC (20s)
@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
def fetch_alert_incidents(_api_client, configuration_ids, start_time_iso):
    """Obtiene incidentes de alerta para IDs de configuración específicos."""
    if not _api_client or not configuration_ids:
        return None
    return _api_client.get_alert_incidents(configuration_ids, start_time_iso)

@st.cache_data
def get_vehicle_config_ids(all_configs, vehicle_id):
    vehicle_config_ids = []
    vehicle_id_str = str(vehicle_id)

    if not all_configs or 'data' not in all_configs:
        return []

    for config in all_configs['data']:
        target = config.get('target')
        if not target:
            continue
            
        asset_ids = target.get('assetIds', [])
        asset_ids_str = [str(aid) for aid in asset_ids]
        
        if vehicle_id_str in asset_ids_str:
            vehicle_config_ids.append(config['id'])
            
    print(f"Encontradas {len(vehicle_config_ids)} configs de alerta para el vehículo {vehicle_id_str}: {vehicle_config_ids}")
    return vehicle_config_ids


# --- 4. FUNCIONES DE RENDERIZADO (CACHEADAS) ---

def render_vehicle_info_and_sensors(vehicle_obj, sensor_config):
    if not vehicle_obj:
        st.sidebar.error("No se ha seleccionado ningún vehículo.")
        return
        
    st.sidebar.markdown(f"<div class='vehicle-info-header'>Vehículo</div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<div class='vehicle-info-value'>{vehicle_obj.get('name', 'N/A')}</div>", unsafe_allow_html=True)
    
    if vehicle_obj.get('serial'):
        st.sidebar.markdown(f"<div class='vehicle-info-header' style='margin-top: 5px;'>Serial</div>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<div class='vehicle-info-value'>{vehicle_obj.get('serial')}</div>", unsafe_allow_html=True)
    
    st.sidebar.markdown(f"<div class='vehicle-info-header' style='margin-top: 5px;'>ID</div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<div class='vehicle-info-value'>{vehicle_obj.get('id')}</div>", unsafe_allow_html=True)
    
    st.sidebar.markdown(f"<div class='sensor-list-header' style='margin-top: 10px;'>Dispositivos Pareados</div>", unsafe_allow_html=True)
    
    if not sensor_config:
        st.sidebar.caption("No hay 'sensorConfiguration' para este vehículo.")
        st.sidebar.markdown("<div class='sensor-separator'></div>", unsafe_allow_html=True)
        return

    sensors = {}
    for area in sensor_config.get('areas', []):
        for s in area.get('temperatureSensors', []):
            if s['id'] not in sensors: sensors[s['id']] = {'name': s['name'], 'type': '🌡️ Temp'}
        for s in area.get('humiditySensors', []):
            if s['id'] not in sensors: 
                sensors[s['id']] = {'name': s['name'], 'type': '💧 Humedad'}
            else:
                sensors[s['id']]['type'] = '🌡️💧 Temp/Hum'
                
    for door in sensor_config.get('doors', []):
        s = door.get('sensor')
        if s and s['id'] not in sensors:
             sensors[s['id']] = {'name': s['name'], 'type': '🚪 Puerta'}

    if not sensors:
        st.sidebar.caption("No hay sensores EM/DM pareados.")
    else:
        for id, data in sensors.items():
            model = "EM21/EM22" if 'Temp' in data['type'] else "DM11"
            st.sidebar.markdown(f"<span class='sensor-list-item'>({model}) {data['name']}<br>&nbsp;&nbsp;↳ {id}</span>", unsafe_allow_html=True)
    
    st.sidebar.markdown("<div class='sensor-separator'></div>", unsafe_allow_html=True)


def parse_fault_code(fault_time, fault_obj, code_type):
    codes_found = []
    
    if code_type == "obdii":
        if not fault_obj or 'diagnosticTroubleCodes' not in fault_obj:
            return []
        
        for module in fault_obj.get('diagnosticTroubleCodes', []):
            for dtc in module.get('confirmedDtcs', []):
                codes_found.append({
                    "Hora": fault_time, "Tipo": "OBD-II Confirmado",
                    "Codigo": dtc.get('dtcShortCode', 'N/A'),
                    "Desc": dtc.get('dtcDescription', 'N/A')
                })
    
    elif code_type == "j1939":
        if not fault_obj or 'diagnosticTroubleCodes' not in fault_obj:
            return []
            
        for dtc in fault_obj.get('diagnosticTroubleCodes', []):
            codes_found.append({
                "Hora": fault_time, "Tipo": "J1939",
                "Codigo": f"SPN {dtc.get('spnId', 'N/A')} FMI {dtc.get('fmiId', 'N/A')}",
                "Desc": dtc.get('spnDescription', 'N/A')
            })
            
    return codes_found

def render_fault_codes(fault_codes_history_list, live_fault_codes_obj):
    st.sidebar.markdown(f"<div class='sensor-list-header'>🛠️ Códigos de Falla (Últimas 24h)</div>", unsafe_allow_html=True)
    
    check_engine_light = False
    if live_fault_codes_obj and live_fault_codes_obj.get('obdii'):
        check_engine_light = live_fault_codes_obj['obdii'].get('checkEngineLightIsOn', False)
        
    if check_engine_light:
        st.sidebar.error("Check Engine Light: ENCENDIDA")
    else:
        st.sidebar.success("Check Engine Light: Apagada")

    if not fault_codes_history_list:
        st.sidebar.caption("No hay historial de códigos de falla en las últimas 24h.")
        st.sidebar.markdown("<div class='sensor-separator'></div>", unsafe_allow_html=True)
        return

    fault_list_for_df = []
    for fault_event in reversed(fault_codes_history_list): # Más recientes primero
        fault_time = pd.to_datetime(fault_event.get('time')).tz_convert(MEXICO_TZ).strftime('%H:%M')
        
        obdii_codes = parse_fault_code(fault_time, fault_event.get('obdii'), "obdii")
        fault_list_for_df.extend(obdii_codes)
        
        j1939_codes = parse_fault_code(fault_time, fault_event.get('j1939'), "j1939")
        fault_list_for_df.extend(j1939_codes)

    if fault_list_for_df:
        df_faults = pd.DataFrame(fault_list_for_df)
        st.sidebar.dataframe(df_faults, hide_index=True)
    else:
        st.sidebar.caption("No se encontraron códigos de falla activos en las últimas 24h.")
        
    st.sidebar.markdown("<div class='sensor-separator'></div>", unsafe_allow_html=True)


# --- v56: FUNCIÓN DE GRÁFICA CORREGIDA ---
@st.cache_data
def render_mini_chart(series_data, color):
    """
    (v56) Renderiza el mini-gráfico.
    - FIX: 'shape' (hv) y 'range' (y-axis) se aplican SÓLO si los
      valores únicos están entre 0 y 1 (es decir, es la puerta).
    """
    if series_data is None or series_data.empty or series_data.isna().all():
        # Mostrar un gráfico vacío pero con formato
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark", height=70, margin=dict(t=5, b=5, l=5, r=5),
            xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False,
            annotations=[dict(text="No hay datos históricos", xref="paper", yref="paper",
                              x=0.5, y=0.5, showarrow=False, font=dict(size=10, color="#888"))]
        )
        config = {'displayModeBar': False}
        st.plotly_chart(fig, config=config, use_container_width=True)
        return
        
    fig = go.Figure()
    
    # v56: Comprobar si es un gráfico de puerta (solo valores 0 y 1)
    unique_vals = series_data.unique()
    # Comprobar que todos los valores únicos (sin NaN) estén en el conjunto {0.0, 1.0}
    is_door_chart = all(v in [0.0, 1.0] for v in unique_vals if pd.notna(v))
    
    # v56: 'shape' es 'hv' (step chart) SOLO si es la puerta.
    shape = 'hv' if is_door_chart else 'linear'
    
    fig.add_trace(go.Scatter(
        x=series_data.index, y=series_data,
        mode='lines', fill='tozeroy',
        line=dict(color=color, width=2, shape=shape), # <-- v56: shape es dinámico
        fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)'
    ))
    
    # v56: Ajustar el rango Y SÓLO si es la puerta.
    if is_door_chart:
        fig.update_yaxes(range=[-0.1, 1.1]) # Rango de 0 a 1 con padding
        
    fig.update_layout(
        template="plotly_dark",
        height=70,
        margin=dict(t=5, b=5, l=5, r=5),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    config = {'displayModeBar': False}
    # v58: Corregido el warning. 'use_container_width' es correcto para plotly_chart
    st.plotly_chart(fig, config=config, use_container_width=True)


@st.cache_data
def render_history_charts(df, title_suffix):
    temp_chart_placeholder = st.empty()
    hum_chart_placeholder = st.empty()
    
    # FIX: Configuración para eliminar advertencia de Plotly
    plotly_config = {'displayModeBar': False}

    if not df.empty and 'temperature' in df.columns and df['temperature'].notna().any():
        fig_temp = px.line(df, y='temperature', labels={'temperature': 'Temperatura'})
        fig_temp.update_layout(
            title=f"Temperatura {title_suffix}",
            template="plotly_dark", height=200,
            xaxis_title="Hora (24h)", 
            xaxis_tickformat='%H:%M',
            yaxis_title="Temp °C",
            margin=dict(t=40, b=40, l=0, r=0),
            hovermode="x unified"
        )
        # v58: Corregido el warning. 'use_container_width' es correcto para plotly_chart
        temp_chart_placeholder.plotly_chart(fig_temp, config=plotly_config, use_container_width=True)
    else:
        temp_chart_placeholder.info(f"No hay datos de temperatura disponibles ({title_suffix}).")

    if not df.empty and 'humidity' in df.columns and df['humidity'].notna().any():
        fig_hum = px.line(df, y='humidity', labels={'humidity': 'Humedad'})
        fig_hum.update_layout(
            title=f"Humedad {title_suffix}",
            template="plotly_dark", height=200,
            xaxis_title="Hora (24h)", 
            xaxis_tickformat='%H:%M',
            yaxis_title="Humedad %",
            margin=dict(t=40, b=40, l=0, r=0),
            hovermode="x unified"
        )
        # v58: Corregido el warning. 'use_container_width' es correcto para plotly_chart
        hum_chart_placeholder.plotly_chart(fig_hum, config=plotly_config, use_container_width=True)
    else:
        hum_chart_placeholder.info(f"No hay datos de humedad disponibles ({title_suffix}).")
    
    return df

# Esta función es la nueva forma de llamar a la lógica de alerta.
# Se le da un 'key' que Streamlit usa para determinar si debe reejecutar la función.
@st.cache_data(ttl=8) # Refresco cada 8 segundos
def run_alert_log(database_url_key):
    """
    Función de contenedor no-asíncrona para la lógica asíncrona de alertas.
    """
    async def fetch_alerts():
        """
        Función asíncrona real que interactúa con la base de datos.
        """
        if not database.is_connected:
            await database.connect()
            
        # Consulta de alertas
        query = alerts.select().order_by(alerts.c.timestamp.desc()).limit(15)
        results = await database.fetch_all(query)
        
        # --- FIX CRÍTICO: CONVERTIR A LISTA DE DICCIONARIOS SERIALIZABLES ---
        # (v65) a_dict() es necesario para el nuevo tipo de resultado de SQLAlchemy 2.0
        serializable_results = [dict(row._mapping) for row in results]
        
        # FIX: Desconexión obligatoria para liberar el lock de conexión en el pool
        try:
            await database.disconnect() 
        except Exception as e:
            # Ignorar errores si la desconexión falla (ej. ya estaba desconectada)
            print(f"Advertencia al desconectar DB: {e}") 
        
        return serializable_results

    try:
        results = asyncio.run(fetch_alerts())
        return results
    except Exception as e:
        print(f"Error en run_alert_log (fetch_alerts): {e}")
        return []

# --- (MODIFICADO v65/v66) ---
def render_alert_log_section():
    """
    Renderiza la sección de Alertas en Vivo (Webhooks)
    """
    
    results = run_alert_log(DATABASE_URL)

    if not results:
        st.info("No se han registrado alertas (webhooks) en la base de datos.")
        return

    # Lógica de notificación (basada en el ID del evento, no la tabla)
    latest_alert_id = results[0]['event_id']
    
    if st.session_state.last_alert_id is None:
        st.session_state.last_alert_id = latest_alert_id
    
    elif latest_alert_id != st.session_state.last_alert_id:
        alert_msg = f"¡Nueva Alerta: {results[0]['alert_type']} en {results[0]['vehicle_name']}!"
        st.error(alert_msg, icon="🚨")
        # --- FIX: Reproducción de audio robusta (HTML injection) ---
        audio_url = "https://cdn.pixabay.com/audio/22/03/15/audio_2210e72c83.mp3" # Sonido corto de alerta
        st.markdown(
            f"""
            <audio autoplay controls style="display:none;">
                <source src="{audio_url}" type="audio/mp3">
            </audio>
            """,
            unsafe_allow_html=True
        )
        st.session_state.last_alert_id = latest_alert_id

    # Preparar DataFrame (Hora, Referencia, Tipo de alerta, Incidente)
    data_for_df = []
    for row in results:
        timestamp_utc = row['timestamp']
        if timestamp_utc:
            if timestamp_utc.tzinfo is None:
                 timestamp_utc = timestamp_utc.replace(tzinfo=pytz.utc)
            timestamp_local = timestamp_utc.astimezone(MEXICO_TZ).strftime('%H:%M:%S')
        else:
            timestamp_local = "N/A"
        
        data_for_df.append({
            "Hora": timestamp_local,
            "Referencia": row['vehicle_name'],     # (REQUISITO v65) Columna renombrada
            "Tipo de alerta": row['alert_type'],
            "Incidente": row.get('incident_url')   # (REQUISITO v65) Nueva columna
        })
    
    df_alerts = pd.DataFrame(data_for_df)
    
    # (REQUISITO v65/v66) Usar st.data_editor para mostrar el link clickeable
    st.data_editor(
        df_alerts, 
        hide_index=True, 
        use_container_width=False, # <-- (MODIFICACIÓN v66) Ajustar ancho
        disabled=True, # Hacer la tabla de solo lectura
        column_config={
            "Incidente": st.column_config.LinkColumn(
                "Incidente",
                display_text="Ver Incidente", # Texto que se muestra en el link
                help="Click para abrir la URL del incidente en Samsara"
            ),
            "Referencia": st.column_config.TextColumn(
                "Referencia",
                width="medium" # Ajustar ancho
            ),
            "Tipo de alerta": st.column_config.TextColumn(
                "Tipo de alerta",
                width="large", # Ajustar ancho
                # --- (MODIFICACIÓN v66) ---
                # Trunca el texto si es muy largo para mantener la tabla compacta
                max_chars=40, 
                help="El tipo de alerta (truncado a 40 caracteres si es muy largo)"
                # --- (FIN MODIFICACIÓN v66) ---
            ),
             "Hora": st.column_config.TextColumn(
                "Hora",
                width="small" # Ajustar ancho
            )
        },
        # --- (MODIFICACIÓN v66) Nuevo orden de columnas ---
        column_order=("Tipo de alerta", "Incidente", "Hora", "Referencia")
    )


@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
def generate_map_data(live_stats, vehicle_name):
    """Usa el 'live_stats' (snapshot) para el mapa."""
    if live_stats and live_stats.get('gps'):
        gps_data = live_stats['gps']
        lat, lon = gps_data.get('latitude'), gps_data.get('longitude')
        if lat and lon:
            map_center = [lat, lon]
            gps_time_str = pd.to_datetime(gps_data.get('time')).tz_convert(MEXICO_TZ).strftime('%H:%M:%S')
            popup_html = f"{vehicle_name}<br>Vel: {gps_data.get('speedMilesPerHour', 0) * 1.609:.1f} km/h<br>Hora: {gps_time_str}"
            return map_center, popup_html
    return [20.6736, -103.344], "No hay datos de GPS."

@st.cache_data
def get_ia_prediction(_ai_model, temperature_series, step_sec_ai):
    if not temperature_series.empty and len(temperature_series) >= 5:
        return _ai_model.get_temperature_forecast(
            temperature_series,
            steps_ahead=12,
            step_seconds=step_sec_ai
        )
    return None, []

# --- 5. INICIALIZACIÓN DE LA APP ---

vehicle_names, vehicle_map_obj = load_vehicle_list(st.session_state.api_client)

if not vehicle_names:
    st.error("No se pudieron cargar los vehículos. ¿La clave de API es correcta y tiene permisos?")
    st.stop()

# --- 6. BARRA LATERAL (SIDEBAR) ---

st.sidebar.title("Panel de Control")

def on_vehicle_change():
    st.session_state.selected_vehicle_obj = vehicle_map_obj.get(st.session_state.selected_vehicle_name)
    if st.session_state.selected_vehicle_obj:
        st.session_state.sensor_config = st.session_state.selected_vehicle_obj.get('sensorConfiguration')
    else:
        st.session_state.sensor_config = None
    st.cache_data.clear()


if st.session_state.selected_vehicle_name is None and vehicle_names:
    st.session_state.selected_vehicle_name = vehicle_names[0]
    on_vehicle_change()

st.sidebar.selectbox(
    "Selecciona un Vehículo:",
    vehicle_names,
    key='selected_vehicle_name',
    on_change=on_vehicle_change,
    label_visibility="collapsed"
)

if not st.session_state.selected_vehicle_obj:
    st.error("Error al cargar datos del vehículo.")
    st.stop()

# --- 7. CARGA DE DATOS PRINCIPAL (CACHEADA) ---
selected_vehicle_id = st.session_state.selected_vehicle_obj['id']
selected_vehicle_name = st.session_state.selected_vehicle_obj['name']
sensor_config = st.session_state.sensor_config

# 1. SNAPSHOT (Para Mapa y Check Engine)
live_stats = fetch_live_kpis(
    st.session_state.api_client, sensor_config, selected_vehicle_id
)
live_fault_codes_obj = live_stats.get('faultCodes') if live_stats else None


# 2. HISTORIAL DE SENSORES (1h, 30s) -> Para KPIs de Puerta/Temp/Hum
history_results_1h, column_map_1h = fetch_live_sensor_history(
    st.session_state.api_client, sensor_config, 60, 30
)
df_history_1h = process_sensor_history_data(
    history_results_1h, column_map_1h, step_seconds=30
)

# 2.5. (v43) HISTORIAL DE VEHÍCULO (1h) -> Para KPI de Batería EN VIVO
raw_stats_history_1h = fetch_vehicle_stats_history(
    st.session_state.api_client, selected_vehicle_id, 60 # 60 minutos
)
df_battery_history_1h, _ = process_vehicle_stats_history(raw_stats_history_1h)


# 3. HISTORIAL DE SENSORES (24h, 10min) -> Para Gráficos de Tendencia
history_results_24h, column_map_24h = fetch_live_sensor_history(
    st.session_state.api_client, sensor_config, 1440, 600
)
df_history_24h = process_sensor_history_data(
    history_results_24h, column_map_24h, step_seconds=600
)

# 4. (v37) HISTORIAL DE VEHÍCULO (24h) -> Para Gráfico de Batería y Lista de Fallas
raw_stats_history_24h = fetch_vehicle_stats_history(
    st.session_state.api_client, selected_vehicle_id, 1440 # 1440 minutos = 24h
)
df_battery_history_24h, all_fault_codes_list = process_vehicle_stats_history(raw_stats_history_24h)

# 5. (v42) CARGA DE ALERTAS REALES (API) - Para KPI de Puerta
all_alert_configs = load_alert_configurations(st.session_state.api_client)
vehicle_config_ids = get_vehicle_config_ids(all_alert_configs, selected_vehicle_id)
start_time_alerts = datetime.now(pytz.utc) - timedelta(hours=24) # v59: 24h para buscar cambios
alert_incidents = fetch_alert_incidents(
    st.session_state.api_client, 
    vehicle_config_ids, 
    start_time_alerts.isoformat()
)

# 6. (v44) PROCESAR ALERTA DE PUERTA (API)
latest_door_alert = None
if alert_incidents and 'data' in alert_incidents:
    # Ordenar por 'startedAt' descendente para obtener la más reciente primero
    sorted_alerts = sorted(
        alert_incidents['data'], 
        key=lambda x: x.get('startedAt', '1970-01-01T00:00:00Z'), 
        reverse=True
    )
    for alert in sorted_alerts: 
        alert_name_lower = alert.get('name', '').lower()
        if 'door' in alert_name_lower or 'puerta' in alert_name_lower:
            latest_door_alert = {
                "name": alert.get('name'),
                "time": pd.to_datetime(alert.get('startedAt')).tz_convert(MEXICO_TZ),
                "details": alert.get('details', {}).get('description', '')
            }
            break 

# --- 8. RENDERIZADO DE LA BARRA LATERAL ---
render_vehicle_info_and_sensors(st.session_state.selected_vehicle_obj, sensor_config)
render_fault_codes(all_fault_codes_list, live_fault_codes_obj)


# --- 9. RENDERIZADO DEL CUERPO PRINCIPAL ---

# --- FILA DE KPIs (ACTUALIZADA CADA 30S) ---
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

# FIX: Altura de contenedor unificada a 200px.
KPI_HEIGHT = 200

# --- v63: KPI DE PUERTA (Refactor con HTML/CSS) ---
with kpi_col1:
    with st.container(border=True, height=KPI_HEIGHT):
        
        # --- 1. Inicializar variables ---
        door_val_str, door_emoji = "N/A", "🚪"
        door_color = "#333333" # Gris oscuro por defecto
        
        # Textos para el detalle
        telemetria_time_str = "Últ. telemetría: N/A"
        prev_event_time_str = "Evento previo: N/A"
        
        df_door_1h = df_history_1h.get('doorClosed')
        
        if df_door_1h is not None and not df_door_1h.empty:
            # --- 2. Obtener estado actual ---
            latest_door_status = df_door_1h.iloc[-1]
            latest_door_time = df_door_1h.index[-1]
            # (Requisito) Hora de la última telemetría
            telemetria_time_str = f"Últ. telemetría: {latest_door_time.strftime('%H:%M:%S')}"
            
            # Definir estado actual, color y texto de estado opuesto
            if latest_door_status == 1:
                door_val_str = "Puerta Cerrada" # (Requisito)
                door_emoji = "🔒"
                door_color = "#28a745" # (Requisito) Verde
                opposite_state_str = "abierta"
            else:
                door_val_str = "Puerta Abierta" # (Requisito)
                door_emoji = "🚨"
                door_color = "#dc3545" # (Requisito) Rojo
                opposite_state_str = "cerrada"

            # --- 3. Encontrar el timestamp del ÚLTIMO ESTADO OPUESTO ---
            try:
                # Encontrar todos los eventos DIFERENTES al estado actual
                different_events = df_door_1h[df_door_1h != latest_door_status]
                
                if not different_events.empty:
                    # Obtener el timestamp del ÚLTIMO evento diferente
                    last_opposite_time = different_events.index[-1]
                    # (Requisito) Hora del evento previo
                    prev_event_time_str = f"Evento previo (últ. {opposite_state_str}): {last_opposite_time.strftime('%H:%M')}"
                else:
                    # Si no hay eventos diferentes, ha estado así por >1h
                    prev_event_time_str = f"Evento previo: (hace >1h)"
            except Exception as e:
                print(f"Error al calcular el tiempo de cambio de puerta: {e}")
                prev_event_time_str = "Evento previo: Error"
        
        # --- 4. (v63) Lógica de Alerta API eliminada del KPI ---

        # --- 5. Renderizar el KPI con HTML/CSS ---
        st.markdown(f"""
        <div style="
            background-color: {door_color};
            border-radius: 8px;
            padding: 12px;
            height: 105px; /* Altura fija para el texto, deja espacio para el gráfico */
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        ">
            <div style="
                font-size: 1.2rem; 
                font-weight: bold; 
                color: white; 
                line-height: 1.2;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
            ">
                {door_emoji} {door_val_str}
            </div>
            <div style="
                font-size: 0.75rem; 
                color: #f0f2f6; /* Blanco suave */
                line-height: 1.3;
                margin-top: 8px;
            ">
                {telemetria_time_str}<br>
                {prev_event_time_str}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Renderizar el mini-gráfico (sin cambios)
        render_mini_chart(df_door_1h, "#3498DB")

# --- v60: KPI DE TEMPERATURA (Refactor de consistencia) ---
with kpi_col2:
    with st.container(border=True, height=KPI_HEIGHT):
        temp_val_str, temp_time_str = "N/A", "(N/A)"
        temp_emoji = "🌡️" # v60
        df_temp_1h = df_history_1h.get('temperature')
        
        if df_temp_1h is not None and not df_temp_1h.empty:
            latest_temp = df_temp_1h.iloc[-1]
            latest_temp_time = df_temp_1h.index[-1]
            temp_val_str = f"{latest_temp:.1f} °C"
            temp_time_str = latest_temp_time.strftime('(%H:%M)')
        
        st.metric(label=f"{temp_emoji} Temperatura {temp_time_str}", value=temp_val_str)
        render_mini_chart(df_temp_1h, "#FFA500")

# --- v6D: KPI DE HUMEDAD (Refactor de consistencia) ---
with kpi_col3:
    with st.container(border=True, height=KPI_HEIGHT):
        hum_val_str, hum_time_str = "N/A", "(N/A)"
        hum_emoji = "💧" # v60
        df_hum_1h = df_history_1h.get('humidity')

        if df_hum_1h is not None and not df_hum_1h.empty:
            latest_hum = df_hum_1h.iloc[-1]
            latest_hum_time = df_hum_1h.index[-1]
            hum_val_str = f"{latest_hum:.0f} %"
            hum_time_str = latest_hum_time.strftime('(%H:%M)')
        
        st.metric(label=f"{hum_emoji} Humedad {hum_time_str}", value=hum_val_str)
        # v56: Color cambiado para diferenciarlo de la puerta
        render_mini_chart(df_hum_1h, "#5DADE2")
            
# --- v60: KPI DE BATERÍA (Refactor de consistencia) ---
with kpi_col4:
    with st.container(border=True, height=KPI_HEIGHT):
        bat_val_str, bat_time_str = "N/A", "(N/A)"
        bat_emoji = "🔋" # v60
        df_bat_1h = df_battery_history_1h.get('value')
        
        if df_bat_1h is not None and not df_bat_1h.empty:
            latest_bat = df_bat_1h.iloc[-1]
            latest_bat_time = df_bat_1h.index[-1]
            bat_val_str = f"{latest_bat:.2f} V"
            bat_time_str = latest_bat_time.strftime('(%H:%M)')
        
        st.metric(label=f"{bat_emoji} Batería {bat_time_str}", value=bat_val_str)
        
        # v55: Usar color verde para el gráfico de batería
        render_mini_chart(df_bat_1h, "#2ECC71")


st.markdown("---") 

# --- FILA PRINCIPAL: MAPA (5) + ALERTAS (3) + TENDENCIAS (4) ---

# Proporción (5, 3, 4) para el mapa, alertas y tendencias.
col_map, col_alerts, col_trends = st.columns((5, 3, 4)) 

with col_map:
    # --- MAPA (Usa Snapshot) ---
    st.subheader("📍 Ubicación en Tiempo Real")
    
    map_center, popup_html = generate_map_data(live_stats, selected_vehicle_name)

    m = folium.Map(location=map_center, zoom_start=14, tiles="CartoDB dark_matter")
    if popup_html != "No hay datos de GPS.":
        icon_html = """
        <div style="width: 20px; height: 20px; background-color: #2ECC71; border-radius: 50%;
            border: 2px solid #FFFFFF; box-shadow: 0 0 10px #2ECC71, 0 0 12px #2ECC71;
            animation: pulse 1.5s infinite;"></div>
        <style>@keyframes pulse { 0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(46, 204, 113, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 15px rgba(46, 204, 113, 0); }
            100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(46, 204, 113, 0); } }
        </style>
        """
        folium.Marker(location=map_center, popup=popup_html, icon=DivIcon(html=icon_html)).add_to(m)
    
    # v58: FIX DEPRECATION WARNING.
    # Se eliminan 'returned_objects' y 'use_container_width'.
    # Se usa 'width="100%"' que es el parámetro correcto para st_folium.
    st_folium(m, height=500, width="100%")
    

with col_alerts:
    # --- LOG DE ALERTAS (WEBHOOKS) ---
    #
    # === (MODIFICACIÓN v66) ===
    # Revertido al contenedor original, como lo pediste.
    
    st.subheader("🔔 Alertas en Vivo (Webhooks)")
    
    # Renderiza el log de alertas con el nuevo manejo asíncrono
    try:
        # Se envuelve en un contenedor para que ocupe el espacio vertical
        # La altura del contenedor será determinada por el contenido del mapa/tendencias.
        with st.container(border=True): 
             render_alert_log_section()
    except Exception as e:
        # Esto se muestra si hay un problema al renderizar, por ejemplo si la DB está caída
        st.error(f"Error fatal al renderizar log de alertas: {e}")
        print(f"Error fatal al renderizar log de alertas: {e}")
    

with col_trends:
    # --- GRÁFICOS DE 24H (Cacheados) ---
    st.subheader(f"📈 Tendencias (24h)")
    
    df_ai_input = pd.DataFrame()
    step_sec_ai = 600
    
    # Se renderizan las gráficas de Temperatura/Humedad (cada una tiene height=200 en render_history_charts)
    df_display = render_history_charts(df_history_24h, "(Últimas 24h)")
    
    st.markdown("<br>", unsafe_allow_html=True)
    if not df_battery_history_24h.empty and 'value' in df_battery_history_24h.columns:
        fig_bat_24h = px.line(df_battery_history_24h, y='value', labels={'value': 'Voltaje'})
        fig_bat_24h.update_layout(
            title=f"Batería (Últimas 24h)",
            template="plotly_dark", height=200,
            xaxis_title="Hora (24h)", 
            xaxis_tickformat='%H:%M',
            yaxis_title="Voltaje V",
            margin=dict(t=40, b=40, l=0, r=0),
            hovermode="x unified"
        )
        plotly_config = {'displayModeBar': False}
        # v58: Corregido el warning. 'use_container_width' es correcto para plotly_chart
        st.plotly_chart(fig_bat_24h, config=plotly_config, use_container_width=True)
    else:
        st.info("No hay datos de batería disponibles (Últimas 24h).")

    if 'temperature' in df_display.columns:
        df_ai_input = df_display.dropna(subset=['temperature'])[['temperature']]
        
    st.markdown("---")


# --- FILA INFERIOR: PREDICCIÓN DE IA (ANCHO COMPLETO) ---
st.subheader("🔮 Predicción de Temperatura (IA)") 
with st.container(border=True): 
    forecast_index, forecast_values = get_ia_prediction(
        st.session_state.ai_model, 
        df_ai_input['temperature'] if 'temperature' in df_ai_input else pd.Series(dtype=float), 
        step_sec_ai
    )
    
    if forecast_values is not None and len(forecast_values) > 0:
        df_forecast = pd.DataFrame({'timestamp': forecast_index, 'Predicción': forecast_values}).set_index('timestamp')
        df_real = df_ai_input[['temperature']].rename(columns={'temperature': 'Real'})
        df_plot = pd.concat([df_real, df_forecast])
        
        fig_fc = px.line(df_plot, markers=True, title="Predicción de Temperatura")
        fig_fc.update_layout(template="plotly_dark", yaxis_title="Temperatura °C", margin=dict(t=40, b=0, l=0, r=0))
        plotly_config = {'displayModeBar': False}
        # v58: Corregido el warning. 'use_container_width' es correcto para plotly_chart
        st.plotly_chart(fig_fc, config=plotly_config, use_container_width=True) 
    else:
        st.info(f"Se necesitan al menos 5 puntos de datos para la predicción (actuales: {len(df_ai_input)}).")
