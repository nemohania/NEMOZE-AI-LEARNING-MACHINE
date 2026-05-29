"""
WebSocket connections for real-time updates
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import json
import logging
from datetime import datetime
from typing import Set

logger = logging.getLogger(__name__)
router = APIRouter()

class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {str(e)}")
    
    async def send_personal(self, websocket: WebSocket, message: dict):
        """Send message to specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {str(e)}")

manager = ConnectionManager()

@router.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time model updates"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Process message and send response
            response = {
                "type": "update",
                "timestamp": datetime.utcnow().isoformat(),
                "data": message,
                "status": "received"
            }
            
            await manager.send_personal(websocket, response)
            
            # Optionally broadcast to all clients
            if message.get("broadcast"):
                broadcast_msg = {
                    "type": "broadcast",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": message
                }
                await manager.broadcast(broadcast_msg)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)

@router.websocket("/ws/predictions")
async def predictions_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time predictions"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            
            # Simulate prediction stream
            prediction_update = {
                "type": "prediction",
                "timestamp": datetime.utcnow().isoformat(),
                "prediction": [0.1, 0.2, 0.7],
                "confidence": 0.85
            }
            
            await manager.send_personal(websocket, prediction_update)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Predictions stream error: {str(e)}")
        manager.disconnect(websocket)
