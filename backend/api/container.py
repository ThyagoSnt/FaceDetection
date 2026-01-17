from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Request

from core.face.face_embedder import FaceEmbedder
from core.storage.person_store import PersonStore
from core.camera.camera_scheduler import CameraScheduler


@dataclass
class AppContainer:
    """
    Holds singleton services for the application.
    """
    face_embedder: FaceEmbedder
    person_store: PersonStore
    camera_scheduler: CameraScheduler


def build_container(model_root: str = "./models", cycle_sleep: float = 0.5) -> AppContainer:
    """
    Creates and wires the singletons used by the API.
    """
    face_embedder = FaceEmbedder(model_root=model_root)
    person_store = PersonStore()
    camera_scheduler = CameraScheduler(face_embedder, person_store, cycle_sleep=cycle_sleep)
    return AppContainer(
        face_embedder=face_embedder,
        person_store=person_store,
        camera_scheduler=camera_scheduler,
    )


def get_container(request: Request) -> AppContainer:
    """
    Retrieves the container from app.state.
    """
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise RuntimeError("AppContainer is not initialized on app.state.container")
    return container


def get_face_embedder(container: AppContainer = Depends(get_container)) -> FaceEmbedder:
    return container.face_embedder


def get_person_store(container: AppContainer = Depends(get_container)) -> PersonStore:
    return container.person_store


def get_camera_scheduler(container: AppContainer = Depends(get_container)) -> CameraScheduler:
    return container.camera_scheduler
