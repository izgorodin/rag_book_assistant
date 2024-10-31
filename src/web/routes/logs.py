from fastapi import APIRouter, HTTPException, Depends
from src.utils.logger import LoggerManager
from src.web.auth.dependencies import get_current_user
import os

router = APIRouter()

@router.get("/logs/{log_type}")
async def get_logs(
    log_type: str, 
    lines: int = 100,
    user: str = Depends(get_current_user)
):
    """
    Получить последние строки логов.
    log_type: тип логов (main или rag)
    lines: количество последних строк
    """
    try:
        log_path = os.path.join('logs', 'app.log')
        
        if not os.path.exists(log_path):
            return {"logs": []}
            
        with open(log_path, 'r') as f:
            all_lines = f.readlines()
            return {"logs": all_lines[-lines:]}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
