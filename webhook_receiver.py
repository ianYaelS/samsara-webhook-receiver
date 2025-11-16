# --------------------------------------------------------------------------
# webhook_receiver.py
#
# Servidor de Webhooks (FastAPI) - v71
#
# - (FIX v71): Corregida la lógica de parseo de 'AlertIncident'.
#   - El 'vehicle_id' (para filtrar) ahora se busca en *todas* las
#     alertas, incluido el objeto 'details.vehicle' en los formularios.
#   - El 'vehicle_name' (Referencia) es el 'driver_id' para Formularios
#     y el 'vehicle_name' para otras alertas.
#   - SI Y SOLO SI no se encuentra un 'vehicle_id' en un formulario,
#     se usará el 'driver_id' como 'vehicle_id' de respaldo.
# --------------------------------------------------------------------------

import os
import databases
import sqlalchemy
from sqlalchemy import Column, Integer, String, DateTime, JSON, MetaData
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional
import logging
import hmac
import hashlib
import json
import base64 
from datetime import datetime # <-- IMPORTACIÓN CLAVE

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuración de la Base de Datos y Secretos ---

DATABASE_URL = os.getenv("DATABASE_URL")
SAMSARA_WEBHOOK_SECRET = os.getenv("SAMSARA_WEBHOOK_SECRET")

# Asegurarse de que la URL de Render (postgres://) funcione con sqlalchemy (postgresql://)
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    # FIX: Reemplazar el prefijo postgres:// por postgresql://
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# ROBUSTEZ LOCAL: Si DATABASE_URL no existe (modo dev), usar SQLite persistente.
if not DATABASE_URL:
    logger.warning("ADVERTENCIA: DATABASE_URL no configurada. Usando SQLite local (alerts.db).")
    DATABASE_URL = "sqlite:///alerts.db" 

if not SAMSARA_WEBHOOK_SECRET:
    logger.error("ERROR FATAL: SAMSARA_WEBHOOK_SECRET no está configurado.")

# DECODIFICACIÓN CRÍTICA: Se realiza una vez al inicio.
SECRET_BYTES = None
if SAMSARA_WEBHOOK_SECRET:
    try:
        # Decodificar el secreto de Base64 a bytes (formato requerido por hmac.new)
        SECRET_BYTES = base64.b64decode(SAMSARA_WEBHOOK_SECRET.encode('utf-8'))
        logger.info("Secreto de Webhook decodificado con éxito.")
    except Exception as e:
        logger.error(f"Error al decodificar el secreto Base64: {e}")

# ... (Definiciones de tablas, engine y metadata) ...
database = databases.Database(DATABASE_URL)
metadata = MetaData()

# --- (MODIFICADO v65) ---
alerts = sqlalchemy.Table(
    "alerts",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("event_id", String, unique=True, index=True),
    Column("timestamp", DateTime(timezone=True)),
    Column("vehicle_id", String), # (v71) ID del Vehículo (o Driver ID si no hay vehículo)
    Column("vehicle_name", String), # "Referencia" (Nombre Vehículo o ID Conductor)
    Column("alert_type", String),
    Column("message", JSON),
    Column("incident_url", String, nullable=True),
    Column("raw_json", JSON),
)

engine = sqlalchemy.create_engine(DATABASE_URL)
try:
    metadata.create_all(engine)
    logger.info("Verificación/creación de tabla 'alerts' completada.")
except Exception as e:
    logger.warning(f"No se pudo asegurar la creación de la tabla (posiblemente PostgreSQL). Error: {e}")


app = FastAPI(title="Samsara Webhook Listener")

# --- Modelos Pydantic ---
class WebhookPayload(BaseModel):
    eventId: str
    eventTime: Optional[str] = None
    eventMs: Optional[int] = None
    eventType: str
    orgId: int
    webhookId: str
    data: Optional[Dict[str, Any]] = None 
    event: Optional[Dict[str, Any]] = None 

# --- Función de Verificación de Firma ---

async def verify_signature(request: Request, body: bytes): 
    """
    Verifica la firma v1 de Samsara, accediendo al encabezado X-Samsara-Timestamp
    a través del objeto Request.
    """
    global SECRET_BYTES
    
    if SECRET_BYTES is None: 
        logger.error("SECRET_BYTES no configurado. Permitiendo el paso (solo para desarrollo sin secreto).")
        return True 

    try:
        timestamp = request.headers['x-samsara-timestamp']
        signature_header = request.headers['x-samsara-signature']
    except KeyError as e:
        logger.warning(f"Falta el encabezado requerido: {e}")
        return False

    v1_hash = None
    parts = signature_header.split(',')
    for part in parts:
        if part.strip().startswith('v1='):
            v1_hash = part.strip().split('=')[1]
            break
    
    if not v1_hash:
        logger.warning("Firma v1 no encontrada en la cabecera.")
        return False

    prefix = bytes('v1:' + timestamp + ':', 'utf-8')
    message = prefix + body
    
    computed_hash = hmac.new(
        SECRET_BYTES,
        message,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(v1_hash, computed_hash)


# --- Funciones de la App ---

@app.on_event("startup")
async def startup():
    await database.connect()
    logger.info(f"Conectado a la base de datos (PostgreSQL): {DATABASE_URL.split('@')[-1]}")

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()
    logger.info("Desconectado de la base de datos.")

@app.get("/")
def read_root():
    return {"status": "Samsara Webhook Listener está en línea."}

@app.post("/webhook")
async def receive_webhook(request: Request):
    """
    Endpoint principal para recibir todos los webhooks de Samsara.
    """
    raw_body = await request.body()
    
    if not await verify_signature(request, raw_body):
        if SECRET_BYTES: 
             logger.error("¡FALLO DE VERIFICACIÓN DE FIRMA!")
             raise HTTPException(status_code=403, detail="Firma inválida.")
        else:
             logger.warning("Firma no verificada (SECRET_BYTES no configurado).")

    try:
        payload_dict = json.loads(raw_body)
        payload = WebhookPayload(**payload_dict)
    except Exception as e:
        logger.error(f"Error al parsear el JSON del payload: {e}")
        raise HTTPException(status_code=400, detail="Payload JSON malformado.")

    
    logger.info(f"===== WEBHOOK DE SAMSARA RECIBIDO ({payload.eventType}) =====")

    if payload.eventType == "Ping":
        logger.info("Ping de Samsara recibido. ¡La conexión funciona!")
        return {"status": "success", "ping_received": True}

    # --- (MODIFICADO v71) ---
    if payload.eventType == "AlertIncident" and payload.data:
        try:
            alert_data = payload.data
            conditions = alert_data.get("conditions", [])
            
            if not conditions:
                logger.warning("Alerta recibida pero sin 'conditions'. Omitiendo.")
                return {"status": "success", "alert_ignored": "no_conditions"}

            first_condition = conditions[0]
            
            # --- Variables a popular ---
            alert_type_str = first_condition.get("description", "Alerta Desconocida") 
            details = first_condition.get("details", {}) 
            vehicle_name_str = "N/A" # "Referencia"
            vehicle_id_str = "N/A"   # "ID de Filtro"
            incident_url_str = alert_data.get("incidentUrl")
            # ---

            happened_at_time_str = alert_data.get("happenedAtTime")
            happened_at_time_dt = None
            if happened_at_time_str:
                happened_at_time_dt = datetime.fromisoformat(happened_at_time_str.replace('Z', '+00:00'))
            
            # --- LÓGICA DE PARSEO (v71) ---

            # PASO 1: (v71) Buscar un VEHÍCULO genérico en 'details'.
            # Esto es clave, ya que "Form Submitted" SÍ puede incluir este objeto
            # si la alerta está configurada para monitorear un vehículo.
            if "vehicle" in details and details.get("vehicle"):
                vehicle_data = details["vehicle"]
                vehicle_id_str = vehicle_data.get("id", "N/A")
                vehicle_name_str = vehicle_data.get("name", "N/A") 
                logger.info(f"Parseo v71 (Paso 1): Encontrado vehículo genérico: {vehicle_id_str}")

            # PASO 2: Buscar IDs de vehículo en alertas específicas (Gateway)
            # Esto puede sobrescribir la Referencia si es más específico
            if details.get("gatewayUnplugged") and details["gatewayUnplugged"].get("vehicle"):
                vehicle_data = details["gatewayUnplugged"]["vehicle"]
                vehicle_id_str = vehicle_data.get("id", vehicle_id_str) # Usar ID si no hay uno
                vehicle_name_str = vehicle_data.get("name", "N/A") # Referencia = Vehicle Name
            
            elif details.get("gatewayDisconnected") and details["gatewayDisconnected"].get("vehicle"):
                vehicle_data = details["gatewayDisconnected"]["vehicle"]
                vehicle_id_str = vehicle_data.get("id", vehicle_id_str) # Usar ID si no hay uno
                vehicle_name_str = vehicle_data.get("name", "N/A") # Referencia = Vehicle Name

            # PASO 3: Si es "Form Submitted", *modificar* el tipo de alerta
            # y *sobrescribir* la REFERENCIA (vehicle_name) al Driver ID.
            if alert_type_str == "Form Submitted" and details.get("formSubmitted"):
                form_data = details.get("formSubmitted", {}).get("form", {})
                fields = form_data.get("fields", [])
                emergency_answer = "N/D"
                question_to_find = "¿Tienes una emergencia (accidente, riesgo médico, robo)?" 

                for field in fields:
                    if field.get("label") == question_to_find:
                        if field.get("multipleChoiceValue"):
                            emergency_answer = field["multipleChoiceValue"].get("value", "N/D")
                            break
                
                # (v68) Formato: "Form: [Respuesta]"
                alert_type_str = f"Form: {emergency_answer}"
                
                submitter_info = form_data.get("submittedBy")
                if submitter_info and submitter_info.get("type") == "driver":
                    driver_id = submitter_info.get('id', 'N/A')
                    # (v7al 1) La *Referencia* (vehicle_name) es el Driver ID
                    vehicle_name_str = driver_id 
                    
                    # (FIX v71) Si *aún* no tenemos un vehicle_id (del Paso 1 o 2),
                    # entonces este es un formulario "sin vehículo" y usamos el
                    # driver_id como el ID de filtro.
                    if vehicle_id_str == "N/A":
                        vehicle_id_str = driver_id
                        logger.info(f"Parseo v71 (Paso 3): Formulario sin vehículo. Usando Driver ID ({driver_id}) como ID de filtro.")
                else:
                     if vehicle_name_str == "N/A": # Si no hay vehículo Y no hay driver
                        vehicle_name_str = "Driver ID N/A"
            # --- Fin de Lógica de Parseo ---

            event_to_store = {
                "event_id": payload.eventId,
                "timestamp": happened_at_time_dt, 
                "alert_type": alert_type_str,
                "message": details,
                "vehicle_name": vehicle_name_str,  # "Referencia"
                "vehicle_id": vehicle_id_str,      # "ID de Filtro"
                "incident_url": incident_url_str,
                "raw_json": payload.dict() 
            }
            
            logger.info(f"Insertando Alerta: {alert_type_str} para Ref: {vehicle_name_str} (Filtro ID: {vehicle_id_str}) (URL: {incident_url_str})")
            
            query = alerts.insert().values(event_to_store)
            await database.execute(query)

            return {"status": "success", "data_received": True}

        except Exception as e:
            logger.error(f"Error al procesar AlertIncident: {e}")
            raise HTTPException(status_code=500, detail=f"Error al procesar: {e}")

    logger.info(f"Evento no manejado: {payload.eventType}")
    return {"status": "success", "event_type_unhandled": payload.eventType}

# --- Para ejecutar localmente ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# <-- CORRECCIÓN: El '}' extra ha sido eliminado.
