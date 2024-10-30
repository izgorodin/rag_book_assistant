from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from scipy import stats
from starlette.middleware.base import BaseHTTPMiddleware
from typing import List, Optional
import logging
from starlette import status

logger = logging.getLogger(__name__)

class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(
        self, 
        app,
        public_paths: Optional[List[str]] = None,
        debug: bool = False
    ):
        super().__init__(app)
        self.public_paths = public_paths or []
        self.debug = debug
        
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        
        # Проверяем публичные пути
        if self._is_public_path(path):
            if self.debug:
                logger.debug(f"Public path accessed: {path}")
            return await call_next(request)
            
        # Проверяем токен
        token = request.headers.get('Authorization')
        if not token and not self._check_session(request):
            if self.debug:
                logger.debug(f"Authentication required for path: {path}")
            return self._handle_unauthorized()
            
        return await call_next(request)
    
    def _is_public_path(self, path: str) -> bool:
        return any(
            path.startswith(public_path) 
            for public_path in self.public_paths
        )
    
    def _check_session(self, request: Request) -> bool:
        # Проверка сессии если используется
        return bool(request.session.get("user")) if hasattr(request, "session") else False
    
    def _handle_unauthorized(self):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Not authenticated"},
            headers={"WWW-Authenticate": "Bearer"}
        )
