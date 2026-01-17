from __future__ import annotations

from fastapi import FastAPI

from backend.api.container import build_container
from backend.api.routers import persons, cameras, person, telegram


def create_app() -> FastAPI:
    app = FastAPI(title="Local Camera Feed Analysis")

    # Build and attach singletons
    app.state.container = build_container(model_root="./models", cycle_sleep=0.5)

    # Routers
    app.include_router(persons.router)
    app.include_router(cameras.router)
    app.include_router(person.router)
    app.include_router(telegram.router)

    # Lifecycle
    @app.on_event("startup")
    def on_startup() -> None:
        app.state.container.camera_scheduler.start()
        print("[INFO] FastAPI app started, CameraScheduler running.")

    @app.on_event("shutdown")
    def on_shutdown() -> None:
        app.state.container.camera_scheduler.stop()
        print("[INFO] FastAPI app shutting down.")

    return app


app = create_app()
