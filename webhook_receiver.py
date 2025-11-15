# --------------------------------------------------------------------------
# webhook_receiver.py
#
# Servidor de Webhooks (FastAPI)
#
# 1. Recibe webhooks de Samsara en /webhook.
# 2. Responde al 'Ping' de Samsara.
# 3. (MODIFICADO) Verifica la firma 'X-Samsara-Signature'.
# 4. (MODIFICADO) Guarda Alertas en la tabla 'alerts' de PostgreSQL.
# --------------------------------------------------------------------------

import os
import databases
import sqlalchemy
# (MODIFICADO) Imports añadidos para la nueva tabla y lógica de firma
from sqlalchemy import Column, Integer, String, DateTime, JSON, MetaData
from sqlalchemy.orm import sessionmaker
from fastapi import FastAPI, Request, HTTPException, Response
from pydantic import BaseModel
from typing import Any, Dict, Optional
import datetime
import uvicorn
import logging
# (MODIFICADO) Imports añadidos para verificar la firma
import hmac
import hashlib
import json

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuración de la Base de Datos ---

# Render.com proporciona esta variable de entorno automáticamente
DATABASE_URL = os.getenv("DATABASE_URL")
# (MODIFICADO) Se añade la variable de entorno para el secreto del webhook
SAMSARA_WEBHOOK_SECRET = os.getenv("SAMSARA_WEBHOOK_SECRET")

# Asegurarse de que la URL de Render (postgres://) funcione con sqlalchemy (postgresql://)
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if not DATABASE_URL:
    logger.warning("ADVERTENCIA: DATABASE_URL no está configurada. Usando SQLite en memoria.")
    DATABASE_URL = "sqlite:///:memory:" # Fallback para pruebas locales

# (MODIFICADO) Se añade la validación del secreto
if not SAMSARA_WEBHOOK_SECRET:
    logger.error("ERROR FATAL: SAMSARA_WEBHOOK_SECRET no está configurado.")
    # En un entorno real, podríamos querer que la app falle si esto no está.
    # Por ahora, solo logueamos el error.

database = databases.Database(DATABASE_URL)
# (MODIFICADO) Se cambia 'metadata' a 'MetaData()' para la nueva definición de tabla
metadata = MetaData()

# (MODIFICADO) Definición de la tabla 'alerts' (como se solicitó)
alerts = sqlalchemy.Table(
    "alerts",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("event_id", String, unique=True, index=True),
    # (MODIFICADO) Columna 'timestamp' de tipo DateTime con Timezone (TIMESTAMPZ)
    Column("timestamp", DateTime(timezone=True)),
    Column("vehicle_id", String),
    Column("vehicle_name", String),
    # (MODIFICADO) 'alert_type' almacenará la descripción (ej. "Gateway Unplugged")
    Column("alert_type", String),
    # (MODIFICADO) 'message' almacenará el JSON de 'details'
    Column("message", JSON),
    Column("raw_json", JSON),
)


# Motor de SQLAlchemy (solo para crear la tabla)
engine = sqlalchemy.create_engine(DATABASE_URL)
metadata.create_all(engine)

app = FastAPI(title="Samsara Webhook Listener")

# --- Modelos Pydantic para validación (simplificados) ---

class WebhookData(BaseModel):
    happenedAtTime: Optional[str] = None
    configurationId: Optional[str] = None
    conditions: Optional[list] = None
    # ... otros campos si son necesarios

class WebhookPayload(BaseModel):
    eventId: str
    eventTime: str
    eventType: str
    orgId: int
    webhookId: str
    data: Optional[Dict[str, Any]] = None # Para AlertIncident
    event: Optional[Dict[str, Any]] = None # Para Ping

# --- (MODIFICADO) Función de Verificación de Firma ---

async def verify_signature(body: bytes, signature_header: str, secret: str):
    """
    Verifica la firma v1 de Samsara.
    """
    if not signature_header:
        logger.warning("Firma no encontrada en la cabecera.")
        return False
        
    if not secret:
        logger.error("SAMSARA_WEBHOOK_SECRET no está configurado. No se puede verificar la firma.")
        # Permitir pasar si el secreto no está configurado (para pruebas locales)
        return True 

    try:
        # Extraer el hash v1
        v1_hash = None
        parts = signature_header.split(',')
        for part in parts:
            if part.strip().startswith('v1='):
                v1_hash = part.strip().split('=')[1]
                break
        
        if not v1_hash:
            logger.warning("Firma v1 no encontrada en la cabecera.")
            return False

        # Calcular nuestro hash
        computed_hash = hmac.new(
            secret.encode(),
            body,
            hashlib.sha256
        ).hexdigest()

        # Comparar de forma segura
        return hmac.compare_digest(v1_hash, computed_hash)

    except Exception as e:
        logger.error(f"Error al verificar la firma: {e}")
        return False

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

# (MODIFICADO) El endpoint ahora usa 'Request' para leer el body crudo
@app.post("/webhook")
async def receive_webhook(request: Request):
    """
    Endpoint principal para recibir todos los webhooks de Samsara.
    (MODIFICADO) Ahora verifica la firma y guarda en la tabla 'alerts'.
    """

    # (MODIFICADO) 1. Leer body crudo y verificar firma
    raw_body = await request.body()
    signature_header = request.headers.get('x-samsara-signature')

    if not await verify_signature(raw_body, signature_header, SAMSARA_WEBHOOK_SECRET):
        # (MODIFICADO) Si el secreto está configurado y la firma falla, rechazar
        if SAMSARA_WEBHOOK_SECRET:
             logger.error("¡FALLO DE VERIFICACIÓN DE FIRMA!")
             raise HTTPException(status_code=403, detail="Firma inválida.")
        else:
             logger.warning("Firma no verificada (SAMSARA_WEBHOOK_SECRET no configurado).")

    # (MODIFICADO) 2. Parsear el payload (ahora que el body crudo fue leído)
    try:
        payload_dict = json.loads(raw_body)
        payload = WebhookPayload(**payload_dict)
    except Exception as e:
        logger.error(f"Error al parsear el JSON del payload: {e}")
        raise HTTPException(status_code=400, detail="Payload JSON malformado.")

    
    logger.info(f"===== WEBHOOK DE SAMSARA RECIBIDO ({payload.eventType}) =====")

    # 3. Manejar el 'Ping' de Samsara
    if payload.eventType == "Ping":
        logger.info("Ping de Samsara recibido. ¡La conexión funciona!")
        # (MODIFICADO) Devolver 200 OK con body (como se solicitó)
        return {"status": "success", "ping_received": True}

    # 4. Manejar 'AlertIncident'
    if payload.eventType == "AlertIncident" and payload.data:
        try:
            alert_data = payload.data
            conditions = alert_data.get("conditions", [])
            
            if not conditions:
                logger.warning("Alerta recibida pero sin 'conditions'. Omitiendo.")
                return {"status": "success", "alert_ignored": "no_conditions"}

            # Extraer info del primer 'condition'
            first_condition = conditions[0]
            
            # (MODIFICADO) Extraer campos según lo solicitado
            description = first_condition.get("description", "Sin descripción") # -> alert_type
            details = first_condition.get("details", {}) # -> message (JSON)
            happened_at_time = alert_data.get("happenedAtTime") # -> timestamp
            
            # Intentar obtener info del vehículo (lógica existente)
            vehicle_name = "N/A"
            vehicle_id = "N/A"
            
            gateway_info = details.get("gatewayUnplugged") or details.get("gatewayDisconnected")
            if gateway_info and gateway_info.get("vehicle"):
                vehicle_name = gateway_info["vehicle"].get("name", "N/A")
                vehicle_id = gateway_info["vehicle"].get("id", "N/A")

            # (MODIFICADO) Preparar datos para la tabla 'alerts'
            event_to_store = {
                "event_id": payload.eventId,
                "timestamp": happened_at_time,
                "alert_type": description,
                "message": details,
                "vehicle_name": vehicle_name,
                "vehicle_id": vehicle_id,
                "raw_json": payload.dict() # Guardar todo el payload
            }
            
            logger.info(f"Insertando Alerta: {description} para {vehicle_name}")
            
            # (MODIFICADO) Insertar en la tabla 'alerts'
            query = alerts.insert().values(event_to_store)
            await database.execute(query)

            # (MODIFICADO) Devolver 200 OK con body (como se solicitó)
            return {"status": "success", "data_received": True}

        except Exception as e:
            logger.error(f"Error al procesar AlertIncident: {e}")
            raise HTTPException(status_code=500, detail=f"Error al procesar: {e}")

    logger.info(f"Evento no manejado: {payload.eventType}")
    # (MODIFICADO) Devolver 200 OK con body (como se solicitó)
    return {"status": "success", "event_type_unhandled": payload.eventType}

# --- Para ejecutar localmente ---
if __name__ == "__main__":
    # Uvicorn se ejecutará en el puerto 8000 localmente
    uvicorn.run(app, host="0.0.0.0", port=8000)
