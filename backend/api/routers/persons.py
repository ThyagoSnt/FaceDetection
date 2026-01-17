from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.deps import require_auth
from backend.api.container import get_person_store
from core.storage.person_store import PersonStore


router = APIRouter(
    tags=["persons"],
    dependencies=[Depends(require_auth)],
)


@router.get("/persons")
def list_persons(person_store: PersonStore = Depends(get_person_store)):
    fn = getattr(person_store, "list_persons", None)
    if callable(fn):
        persons = fn()
        return {"count": len(persons), "persons": persons}
    return {"count": 0, "persons": []}