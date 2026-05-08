from app.user.infrastructure.auth_backend import get_current_user_id 
from app.user.presentation.router import router 
 
__all__ = ["router", "get_current_user_id"]