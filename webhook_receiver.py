from fastapi import FastAPI, Request, HTTPException
import uvicorn
import os

# Crea la aplicación de FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    """
    Ruta raíz para verificar que el servicio está vivo.
    """
    return {"message": "Samsara Webhook Receiver está funcionando!"}

@app.post("/webhook")
async def receive_webhook(request: Request):
    """
    Esta es la ruta principal que recibirá las alertas de Samsara.
    """
    try:
        # Obtener el payload (cuerpo) de la solicitud en formato JSON
        data = await request.json()
        
        # Imprimir los datos en los logs de Render para que puedas depurar
        print("===== ALERTA DE SAMSARA RECIBIDA =====")
        print(data)
        print("======================================")
        
        # Samsara envía un evento "Ping" cuando pruebas el webhook
        # Es buena idea manejarlo explícitamente.
        if data.get("eventType") == "Ping":
            print("Ping de Samsara recibido. ¡La conexión funciona!")
            return {"status": "ping_received"}
        
        # --- AQUÍ VA TU LÓGICA ---
        # Puedes agregar lógica para manejar tipos de alertas específicas
        # Por ejemplo, las que configuraste (Form Submitted, Gateway Unplugged, etc.)
        
        # event = data.get("event", {})
        # alert_condition = event.get("alertConditionId")
        
        # if alert_condition == "GatewayUnplugged":
        #     print("¡Alerta de Gateway desconectado!")
        #     # Haz algo aquí...
        # elif alert_condition == "FormSubmitted":
        #     print("Formulario recibido!")
        #     # Haz algo aquí...
            

        # ¡Importante! Responde 200 OK rápido.
        # Samsara necesita una respuesta 2XX para saber que recibiste el webhook.
        return {"status": "success", "data_received": True}

    except Exception as e:
        # Si algo sale mal, imprime el error y devuelve un error HTTP
        print(f"Error procesando el webhook: {e}")
        raise HTTPException(status_code=400, detail=f"Error al procesar el webhook: {e}")

if __name__ == "__main__":
    # Esto permite ejecutar el script localmente para pruebas
    # Render usará el "Start Command" en su lugar.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
