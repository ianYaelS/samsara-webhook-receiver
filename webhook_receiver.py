# --------------------------------------------------------------------------
# webhook_receiver.py
#
# Servidor de Webhooks (FastAPI) - v67 (Lógica v66 confirmada)
#
# - (v66): Modificada la lógica de 'AlertIncident' (Form Submitted):
#   - La 'Referencia' (vehicle_name) ahora es el 'driver_id' (ej. "54432217")
#     en lugar de "Form Submitted by: ID".
# - (v65): Añadida columna 'incident_url' a la base de datos.
# - (v65): Refactorizada la lógica de 'AlertIncident' para manejar
#   múltiples tipos de alertas y almacenar 'incidentUrl'.
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
    Column("vehicle_id", String),
    Column("vehicle_name", String), # <--- Esta columna ahora es "Referencia"
    Column("alert_type", String),
    Column("message", JSON),
    Column("incident_url", String, nullable=True), # <--- NUEVA COLUMNA
    Column("raw_json", JSON),
)

engine = sqlalchemy.create_engine(DATABASE_URL)
try:
    # Intenta crear la tabla (necesario para SQLite local, ignorado por Render/PostgreSQL si ya existe)
    metadata.create_all(engine)
    logger.info("Verificación/creación de tabla 'alerts' completada.")
except Exception as e:
    # Esto es común en Render, donde el motor no tiene permisos de DDL después del build
    logger.warning(f"No se pudo asegurar la creación de la tabla (posiblemente PostgreSQL). Error: {e}")


app = FastAPI(title="Samsara Webhook Listener")

# --- Modelos Pydantic ---
class WebhookPayload(BaseModel):
    eventId: str
    eventTime: Optional[str] = None  # FIX: Ahora opcional para evitar errores 400
    eventMs: Optional[int] = None    # Incluir por si viene en milisegundos
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

    # 1. Extraer Headers requeridos (Timestamp y Signature)
    try:
        timestamp = request.headers['x-samsara-timestamp']
        signature_header = request.headers['x-samsara-signature']
    except KeyError as e:
        logger.warning(f"Falta el encabezado requerido: {e}")
        return False

    # 2. Extraer el hash v1
    v1_hash = None
    parts = signature_header.split(',')
    for part in parts:
        if part.strip().startswith('v1='):
            v1_hash = part.strip().split('=')[1]
            break
    
    if not v1_hash:
        logger.warning("Firma v1 no encontrada en la cabecera.")
        return False

    # 3. Preparar el mensaje para firmar: v1:<timestamp>:<body>
    prefix = bytes('v1:' + timestamp + ':', 'utf-8')
    message = prefix + body
    
    # 4. Calcular nuestro hash usando el secreto decodificado
    computed_hash = hmac.new(
        SECRET_BYTES,
        message,
        hashlib.sha256
    ).hexdigest()

    # 5. Comparar de forma segura (v1 hash vs hash calculado)
    return hmac.compare_digest(v1_hash, computed_hash)


# --- Funciones de la App ---

@app.on_event("startup")
async def startup():
    """Conectarse a la base de datos al iniciar."""
    try:
        await database.connect()
        # Muestra la conexión que se está usando para confirmar el éxito.
        logger.info(f"Conectado a la base de datos (PostgreSQL): {DATABASE_URL.split('@')[-1]}")
    except Exception as e:
        logger.error(f"Error al conectar a la base de datos: {e}")

@app.on_event("shutdown")
async def shutdown():
    """Desconectarse de la base de datos al apagar."""
    await database.disconnect()
    logger.info("Desconectado de la base de datos.")

@app.get("/")
def read_root():
    """Endpoint raíz para verificar que el servicio está vivo."""
    return {"status": "Samsara Webhook Listener está en línea."}

@app.post("/webhook")
async def receive_webhook(request: Request):
    """
    Endpoint principal para recibir todos los webhooks de Samsara.
    """

    # 1. Leer body crudo 
    raw_body = await request.body()
    
    # 2. Verificar firma 
    if not await verify_signature(request, raw_body):
        # Si el secreto está configurado y la firma falla, rechazar
        if SECRET_BYTES: 
             logger.error("¡FALLO DE VERIFICACIÓN DE FIRMA!")
             raise HTTPException(status_code=403, detail="Firma inválida.")
        else:
             logger.warning("Firma no verificada (SECRET_BYTES no configurado).")

    # 3. Parsear el payload 
    try:
        payload_dict = json.loads(raw_body)
        payload = WebhookPayload(**payload_dict)
    except Exception as e:
        logger.error(f"Error al parsear el JSON del payload: {e}")
        raise HTTPException(status_code=400, detail="Payload JSON malformado.")

    
    logger.info(f"===== WEBHOOK DE SAMSARA RECIBIDO ({payload.eventType}) =====")

    # 4. Manejar el 'Ping' de Samsara
    if payload.eventType == "Ping":
        logger.info("Ping de Samsara recibido. ¡La conexión funciona!")
        return {"status": "success", "ping_received": True}

    # 5. Manejar 'AlertIncident' (MODIFICADO v65/v66)
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
            vehicle_name_str = "N/A" # Esta es la "Referencia"
            vehicle_id_str = "N/A"
            # Esta es la línea clave: extrae 'incidentUrl' del payload 'data'
            incident_url_str = alert_data.get("incidentUrl") 
            # ---

            happened_at_time_str = alert_data.get("happenedAtTime")
            happened_at_time_dt = None
            if happened_at_time_str:
                happened_at_time_dt = datetime.fromisoformat(happened_at_time_str.replace('Z', '+00:00'))
            
            # --- LÓGICA DE PARSEO (v65/v66) ---
            
            # CASO 1: Es una alerta de "Form Submitted"
            if alert_type_str == "Form Submitted" and details.get("formSubmitted"):
                form_data = details.get("formSubmitted", {}).get("form", {})
                form_title = form_data.get("title", "Formulario Enviado")
                fields = form_data.get("fields", [])
                emergency_answer = "N/D"
                question_to_find = "¿Tienes una emergencia (accidente, riesgo médico, robo)?" 

                for field in fields:
                    if field.get("label") == question_to_find:
                        if field.get("multipleChoiceValue"):
                            emergency_answer = field["multipleChoiceValue"].get("value", "N/D")
                            break
                
                # (REQUISITO v65) Formatear el tipo de alerta
                alert_type_str = f"Form: {form_title} ({emergency_answer})"
                
                # (REQUISITO v66) Formatear la Referencia (vehicle_name)
                submitter_info = form_data.get("submittedBy")
                if submitter_info and submitter_info.get("type") == "driver":
                    driver_id = submitter_info.get('id', 'N/A')
                    # --- (MODIFICACIÓN v66) ---
                    # La referencia ahora es SÓLO el ID del conductor
                    vehicle_name_str = driver_id 
                    vehicle_id_str = driver_id
                else:
                    # --- (MODIFICACIÓN v66) ---
                    vehicle_name_str = "Driver ID N/A" # Fallback
            
            # CASO 2: Es cualquier otra alerta (Ej. Gateway Unplugged)
            else:
                gateway_info = details.get("gatewayUnplugged") or details.get("gatewayDisconnected")
                if gateway_info and gateway_info.get("vehicle"):
                    # (REQUISITO v65) La referencia es el nombre del vehículo
                    vehicle_name_str = gateway_info["vehicle"].get("name", "N/A")
                    vehicle_id_str = gateway_info["vehicle"].get("id", "N/A")

            # --- Fin de Lógica de Parseo ---

            # Aquí es donde se guarda en la base de datos
            event_to_store = {
                "event_id": payload.eventId,
                "timestamp": happened_at_time_dt, 
                "alert_type": alert_type_str,      # Tipo de alerta (Formateado o Descripción)
                "message": details,                # JSON de detalles
                "vehicle_name": vehicle_name_str,  # "Referencia" (Vehículo o Driver ID)
                "vehicle_id": vehicle_id_str,      # ID (Vehículo o Driver)
                "incident_url": incident_url_str,  # (NUEVO v65) -> Aquí se pasa la URL
                "raw_json": payload.dict() 
            }
            
            logger.info(f"Insertando Alerta: {alert_type_str} para {vehicle_name_str} (URL: {incident_url_str})")
            
            query = alerts.insert().values(event_to_store)
            await database.execute(query)

            # Éxito: Devolver 200 OK
            return {"status": "success", "data_received": True}

        except Exception as e:
            logger.error(f"Error al procesar AlertIncident: {e}")
            # Devolver 500 para indicarle a Samsara que el error fue interno (la base de datos)
            raise HTTPException(status_code=500, detail=f"Error al procesar: {e}")

    # --- (ELIMINADO v65) ---
    # El bloque 'elif payload.eventType == "FormSubmitted"' se eliminó
    # porque el payload real muestra que está anidado en 'AlertIncident'.

    logger.info(f"Evento no manejado: {payload.eventType}")
    return {"status": "success", "event_type_unhandled": payload.eventType}

# --- Para ejecutar localmente ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
