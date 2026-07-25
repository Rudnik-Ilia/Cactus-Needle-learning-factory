from fastapi.routing import APIRouter

from ...schemas import RouteRequest, RouteResponse
from ...service import tool_use as service_tool

router = APIRouter(prefix="/tool")

@router.post("/which", response_model=RouteResponse)
def tool_use(request: RouteRequest) -> RouteResponse:
    return service_tool.tool_use(request)
