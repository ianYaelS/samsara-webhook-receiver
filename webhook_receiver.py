# --------------------------------------------------------------------------
# webhook_receiver.py
#
# Servidor de Webhooks (FastAPI) - CORRECCIÓN DEFINITIVA DE FIRMA
#
# 1. Recibe webhooks de Samsara en /webhook.
# 2. Responde al 'Ping' de Samsara.
# 3. (CORREGIDO) Verifica la firma, pasando el objeto 'request' a la función de verificación.
# 4. Guarda Alertas en la tabla 'alerts' de PostgreSQL.
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

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuración de la Base de Datos y Secretos ---

DATABASE_URL = os.getenv("DATABASE_URL")
SAMSARA_WEBHOOK_SECRET = os.getenv("SAMSARA_WEBHOOK_SECRET")

if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if not DATABASE_URL:
    logger.warning("ADVERTENCIA: DATABASE_URL no está configurada. Usando SQLite en memoria.")
    DATABASE_URL = "sqlite:///:memory:" 

if not SAMSARA_WEBHOOK_SECRET:
    logger.error("ERROR FATAL: SAMSARA_WEBHOOK_SECRET no está configurado.")

# DECODIFICACIÓN CRÍTICA: Se realiza una vez al inicio.
SECRET_BYTES = None
if SAMSARA_WEBHOOK_SECRET:
    try:
        SECRET_BYTES = base64.b64decode(SAMSARA_WEBHOOK_SECRET.encode('utf-8'))
        logger.info("Secreto de Webhook decodificado con éxito.")
    except Exception as e:
        logger.error(f"Error al decodificar el secreto Base64: {e}")

# ... (Definiciones de tablas, engine y metadata) ...
database = databases.Database(DATABASE_URL)
metadata = MetaData()

alerts = sqlalchemy.Table(
    "alerts",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("event_id", String, unique=True, index=True),
    Column("timestamp", DateTime(timezone=True)),
    Column("vehicle_id", String),
    Column("vehicle_name", String),
    Column("alert_type", String),
    Column("message", JSON),
    Column("raw_json", JSON),
)

engine = sqlalchemy.create_engine(DATABASE_URL)
metadata.create_all(engine)

app = FastAPI(title="Samsara Webhook Listener")

# --- Modelos Pydantic (Sin cambios) ---
class WebhookPayload(BaseModel):
    eventId: str
    eventTime: Optional[str] = None  # <--- CAMBIO: AHORA ES OPCIONAL
    eventMs: Optional[int] = None    # <--- AÑADIDO: Incluir eventMs, que suele venir
    eventType: str
    orgId: int
    webhookId: str
    data: Optional[Dict[str, Any]] = None 
    event: Optional[Dict[str, Any]] = None 

# --- Función de Verificación de Firma (CORREGIDA) ---

async def verify_signature(request: Request, body: bytes): 
    """
    Verifica la firma v1 de Samsara, accediendo al encabezado X-Samsara-Timestamp
    a través del objeto Request.
    """
    global SECRET_BYTES
    
    if SECRET_BYTES is None: 
        logger.error("SECRET_BYTES no configurado. Permitiendo el paso (solo para desarrollo).")
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
    # Comparamos solo los valores hexadecimales
    return hmac.compare_digest(v1_hash, computed_hash)


# --- Funciones de la App (MODIFICADO: Llamada a verify_signature) ---

@app.on_event("startup")
async def startup():
    try:
        await database.connect()
        logger.info(f"Conectado a la base de datos: {DATABASE_URL.split('@')[-1]}")
    except Exception as e:
        logger.error(f"Error al conectar a la base de datos: {e}")

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

    # 1. Leer body crudo 
    raw_body = await request.body()
    
    # 2. Verificar firma (Pasamos el objeto request)
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
        # Debe retornar 200 OK
        return {"status": "success", "ping_received": True}

    # 5. Manejar 'AlertIncident'
    if payload.eventType == "AlertIncident" and payload.data:
        try:
            alert_data = payload.data
            conditions = alert_data.get("conditions", [])
            
            if not conditions:
                logger.warning("Alerta recibida pero sin 'conditions'. Omitiendo.")
                return {"status": "success", "alert_ignored": "no_conditions"}

            first_condition = conditions[0]
            
            description = first_condition.get("description", "Sin descripción") 
            details = first_condition.get("details", {}) 
            happened_at_time = alert_data.get("happenedAtTime") 
            
            # Lógica para obtener Vehicle Name/ID
            vehicle_name = "N/A"
            vehicle_id = "N/A"
            
            gateway_info = details.get("gatewayUnplugged") or details.get("gatewayDisconnected")
            if gateway_info and gateway_info.get("vehicle"):
                vehicle_name = gateway_info["vehicle"].get("name", "N/A")
                vehicle_id = gateway_info["vehicle"].get("id", "N/A")

            event_to_store = {
                "event_id": payload.eventId,
                "timestamp": happened_at_time,
                "alert_type": description,
                "message": details,
                "vehicle_name": vehicle_name,
                "vehicle_id": vehicle_id,
                "raw_json": payload.dict() 
            }
            
            logger.info(f"Insertando Alerta: {description} para {vehicle_name}")
            
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
