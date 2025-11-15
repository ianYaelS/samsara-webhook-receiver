# --------------------------------------------------------------------------
# webhook_receiver.py
#
# Servidor de Webhooks (FastAPI) - FINAL
#
# - Incluye fix de decodificación Base64 para X-Samsara-Signature.
# - Incluye fix de Pydantic (eventTime optional).
# - Incluye Fallback de DB a SQLite (alerts.db) para desarrollo local.
# - FIX CRÍTICO: Conversión de string ISO a datetime para inserción en DB.
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
try:
    metadata.create_all(engine)
except Exception as e:
    logger.warning(f"No se pudo crear la tabla (posiblemente PostgreSQL). Error: {e}")


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
        logger.info(f"Conectado a la base de datos: {DATABASE_URL.split('@')[-1]}")
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
            happened_at_time_str = alert_data.get("happenedAtTime") # <-- Se extrae como string

            # --- FIX CRÍTICO: CONVERTIR STRING ISO A DATETIME ---
            happened_at_time_dt = None
            if happened_at_time_str:
                # El formato de Samsara es ISO con Z (UTC). fromisoformat lo maneja.
                happened_at_time_dt = datetime.fromisoformat(happened_at_time_str.replace('Z', '+00:00'))
            
            # Lógica para obtener Vehicle Name/ID
            vehicle_name = "N/A"
            vehicle_id = "N/A"
            
            gateway_info = details.get("gatewayUnplugged") or details.get("gatewayDisconnected")
            if gateway_info and gateway_info.get("vehicle"):
                vehicle_name = gateway_info["vehicle"].get("name", "N/A")
                vehicle_id = gateway_info["vehicle"].get("id", "N/A")

            event_to_store = {
                "event_id": payload.eventId,
                # <-- USAMOS EL OBJETO DATETIME CONVERTIDO -->
                "timestamp": happened_at_time_dt, 
                "alert_type": description,
                "message": details,
                "vehicle_name": vehicle_name,
                "vehicle_id": vehicle_id,
                "raw_json": payload.dict() 
            }
            
            logger.info(f"Insertando Alerta: {description} para {vehicle_name}")
            
            query = alerts.insert().values(event_to_store)
            await database.execute(query)

            # Éxito: Devolver 200 OK
            return {"status": "success", "data_received": True}

        except Exception as e:
            logger.error(f"Error al procesar AlertIncident: {e}")
            # Devolver 500 para indicarle a Samsara que el error fue interno (la base de datos)
            raise HTTPException(status_code=500, detail=f"Error al procesar: {e}")

    logger.info(f"Evento no manejado: {payload.eventType}")
    return {"status": "success", "event_type_unhandled": payload.eventType}

# --- Para ejecutar localmente ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
