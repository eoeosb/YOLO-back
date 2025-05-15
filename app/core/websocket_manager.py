from typing import Dict, Set, List
from fastapi import WebSocket
import asyncio
from datetime import datetime

class WebSocketManager:
    def __init__(self):
        # file_id를 키로 하고 WebSocket 연결 집합을 값으로 하는 딕셔너리
        self.connections: Dict[str, Set[WebSocket]] = {}
        # 메시지 큐
        self.message_queues: Dict[str, List[dict]] = {}
        # 연결 이벤트
        self.connection_events: Dict[str, asyncio.Event] = {}
        
    async def connect(self, websocket: WebSocket, file_id: str):
        await websocket.accept()
        
        if file_id not in self.connections:
            self.connections[file_id] = set()
            self.message_queues[file_id] = []
            self.connection_events[file_id] = asyncio.Event()
            
        self.connections[file_id].add(websocket)
        self.connection_events[file_id].set()  # 연결 완료 표시
        
        # 큐에 있는 메시지 전송
        if file_id in self.message_queues:
            for message in self.message_queues[file_id]:
                try:
                    await websocket.send_json(message)
                except:
                    break
            self.message_queues[file_id] = []  # 큐 비우기
        
    def disconnect(self, websocket: WebSocket, file_id: str):
        if file_id in self.connections:
            self.connections[file_id].discard(websocket)
            if not self.connections[file_id]:
                del self.connections[file_id]
                if file_id in self.message_queues:
                    del self.message_queues[file_id]
                if file_id in self.connection_events:
                    del self.connection_events[file_id]
                
    async def broadcast_progress(self, file_id: str, data: dict):
        # 연결된 WebSocket이 없으면 메시지를 큐에 저장
        if file_id not in self.connections or not self.connections[file_id]:
            if file_id not in self.message_queues:
                self.message_queues[file_id] = []
            self.message_queues[file_id].append(data)
            return
            
        dead_connections = set()
        for connection in self.connections[file_id]:
            try:
                await connection.send_json(data)
            except:
                dead_connections.add(connection)
        
        # 끊어진 연결 제거
        for dead_connection in dead_connections:
            self.connections[file_id].discard(dead_connection)
            
    async def wait_for_connection(self, file_id: str, timeout: float = 10.0) -> bool:
        """
        WebSocket 연결을 기다립니다.
        
        Args:
            file_id: 파일 ID
            timeout: 최대 대기 시간 (초)
            
        Returns:
            bool: 연결 성공 여부
        """
        if file_id not in self.connection_events:
            self.connection_events[file_id] = asyncio.Event()
            
        try:
            await asyncio.wait_for(self.connection_events[file_id].wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False 