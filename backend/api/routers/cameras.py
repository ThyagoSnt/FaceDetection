from __future__ import annotations

from typing import List, Union

from fastapi import APIRouter, Depends

from backend.api.deps import require_auth
from backend.api.schemas import CameraRequisition
from backend.api.container import get_camera_scheduler
from core.camera.camera_scheduler import CameraScheduler


router = APIRouter(
    tags=["cameras"],
    dependencies=[Depends(require_auth)],
)


@router.get("/cameras")
def list_cameras(
    camera_scheduler: CameraScheduler = Depends(get_camera_scheduler),
):
    """
    Lists all persisted camera jobs.
    """
    cameras = camera_scheduler.list_cameras_public()
    return {"count": len(cameras), "cameras": cameras}


@router.get("/cameras/{camera_id}/status")
def camera_status(
    camera_id: str,
    camera_scheduler: CameraScheduler = Depends(get_camera_scheduler),
):
    """
    Returns the job status for a given camera_id.
    """
    cfg = camera_scheduler.get_camera(camera_id)
    if cfg is None:
        return {"camera_id": camera_id, "status": "NOT_FOUND"}

    return {"camera_id": camera_id, "status": camera_scheduler.get_status(camera_id)}

@router.delete("/cameras/{camera_id}")
def delete_camera(
    camera_id: str,
    camera_scheduler: CameraScheduler = Depends(get_camera_scheduler),
):
    """
    Deletes a camera job from the scheduler and from SQLite.
    """
    ok = camera_scheduler.remove_camera(camera_id)
    return {"camera_id": camera_id, "deleted": bool(ok)}


@router.post("/cameras/schedule")
def schedule_cameras(
    args: Union[CameraRequisition, List[CameraRequisition]],
    camera_scheduler: CameraScheduler = Depends(get_camera_scheduler),
):
    """
    Schedules one or more camera jobs.
    """
    if not isinstance(args, list):
        requests_list: List[CameraRequisition] = [args]
    else:
        requests_list = args

    results = []
    for req in requests_list:
        camera_id = camera_scheduler.add_camera(
            rtsp_link=req.rtsp_link,
            camera_nickname=req.camera_nickname,
            telegram_api_token=req.telegram_api_token,
            telegram_chat_id=req.telegram_chat_id,
            threshold=req.threshold,
            cooldown_seconds=req.cooldown_seconds,
        )
        results.append({"camera_nickname": req.camera_nickname, "camera_id": camera_id})

    return {"count": len(results), "jobs": results}
