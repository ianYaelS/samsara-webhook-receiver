# --------------------------------------------------------------------------
# webhook_receiver.py
#
# Servidor de Webhooks (FastAPI)
#
# 1. Recibe webhooks de Samsara en /webhook.
# 2. Responde al 'Ping' de Samsara.
# 3. Guarda TODOS los 'AlertIncident' en la base de datos PostgreSQL.
# --------------------------------------------------------------------------

import os
import databases
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional
import datetime
import uvicorn
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuración de la Base de Datos ---

# Render.com proporciona esta variable de entorno automáticamente
DATABASE_URL = os.getenv("DATABASE_URL")

# Asegurarse de que la URL de Render (postgres://) funcione con sqlalchemy (postgresql://)
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if not DATABASE_URL:
    logger.warning("ADVERTENCIA: DATABASE_URL no está configurada. Usando SQLite en memoria.")
    DATABASE_URL = "sqlite:///:memory:" # Fallback para pruebas locales

database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

# Definición de la tabla para guardar los webhooks
webhook_events = sqlalchemy.Table(
    "webhook_events",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    sqlalchemy.Column("event_id", sqlalchemy.String, unique=True, index=True),
    sqlalchemy.Column("event_time", sqlalchemy.DateTime, default=datetime.datetime.utcnow),
    sqlalchemy.Column("alert_name", sqlalchemy.String),
    sqlalchemy.Column("description", sqlalchemy.String),
    sqlalchemy.Column("vehicle_name", sqlalchemy.String),
    sqlalchemy.Column("vehicle_id", sqlalchemy.String),
    sqlalchemy.Column("raw_data", sqlalchemy.JSON),
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
async def receive_webhook(payload: WebhookPayload):
    """
    Endpoint principal para recibir todos los webhooks de Samsara.
    """
    
    logger.info(f"===== WEBHOOK DE SAMSARA RECIBIDO ({payload.eventType}) =====")

    # 1. Manejar el 'Ping' de Samsara
    if payload.eventType == "Ping":
        logger.info("Ping de Samsara recibido. ¡La conexión funciona!")
        return {"status": "success", "ping_received": True}

    # 2. Manejar 'AlertIncident'
    if payload.eventType == "AlertIncident" and payload.data:
        try:
            alert_data = payload.data
            conditions = alert_data.get("conditions", [])
            
            if not conditions:
                logger.warning("Alerta recibida pero sin 'conditions'. Omitiendo.")
                return {"status": "success", "alert_ignored": "no_conditions"}

            # Extraer info del primer 'condition' (como en tu ejemplo)
            first_condition = conditions[0]
            description = first_condition.get("description", "Sin descripción")
            
            # Intentar obtener info del vehículo
            vehicle_name = "N/A"
            vehicle_id = "N/A"
            details = first_condition.get("details", {})
            
            # Buscar info del vehículo en 'gatewayUnplugged' o 'gatewayDisconnected'
            gateway_info = details.get("gatewayUnplugged") or details.get("gatewayDisconnected")
            if gateway_info and gateway_info.get("vehicle"):
                vehicle_name = gateway_info["vehicle"].get("name", "N/A")
                vehicle_id = gateway_info["vehicle"].get("id", "N/A")

            # Preparar datos para la base de datos
            event_to_store = {
                "event_id": payload.eventId,
                "event_time": payload.eventTime,
                "alert_name": alert_data.get("configurationId", "N/A"), # Podrías mapear esto a nombres
                "description": description,
                "vehicle_name": vehicle_name,
                "vehicle_id": vehicle_id,
                "raw_data": payload.dict() # Guardar todo el payload
            }
            
            logger.info(f"Insertando Alerta: {description} para {vehicle_name}")
            
            # Insertar en la base de datos
            query = webhook_events.insert().values(event_to_store)
            await database.execute(query)

            return {"status": "success", "data_received": True}

        except Exception as e:
            logger.error(f"Error al procesar AlertIncident: {e}")
            raise HTTPException(status_code=500, detail=f"Error al procesar: {e}")

    logger.info(f"Evento no manejado: {payload.eventType}")
    return {"status": "success", "event_type_unhandled": payload.eventType}

# --- Para ejecutar localmente ---
if __name__ == "__main__":
    # Uvicorn se ejecutará en el puerto 8000 localmente
    uvicorn.run(app, host="0.0.0.0", port=8000)
