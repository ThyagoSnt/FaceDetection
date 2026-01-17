# src/camera_scheduler.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
from uuid import uuid4
import os
import sqlite3
import threading
import time
from datetime import datetime

import pytz

from ..face.face_embedder import FaceEmbedder
from ..storage.person_store import PersonStore, MatchResult
from ..telegram.telegram import TelegramClient


@dataclass
class CameraConfig:
    camera_id: str
    rtsp_link: str
    threshold: float
    telegram_api_token: str
    telegram_chat_id: str
    camera_nickname: str
    cooldown_seconds: int = 3600


class CameraScheduler(threading.Thread):
    def __init__(
        self,
        face_embedder: FaceEmbedder,
        person_store: PersonStore,
        cycle_sleep: float = 0.5,
        db_path: str = "/data/cameras.sqlite",
    ) -> None:
        super().__init__(daemon=True)
        self.face_embedder = face_embedder
        self.person_store = person_store
        self.cycle_sleep = cycle_sleep

        # Lock for in-memory structures
        self._lock = threading.Lock()

        # Dedicated DB lock to avoid interleaving DB operations across threads
        self._db_lock = threading.Lock()

        self._running = True

        # In-memory registry (loaded from DB on startup)
        self._cameras: Dict[str, CameraConfig] = {}

        # key = (camera_id, person_id) -> last notification timestamp
        # Note: this remains in-memory (cooldown resets on restart).
        self._last_notified: Dict[Tuple[str, str], float] = {}

        # SQLite connection / schema
        self.db_path = db_path
        self._conn = self._open_db(self.db_path)
        self._ensure_schema()

        # Load persisted cameras from SQLite so they survive restarts
        self._load_cameras_from_db()

    # SQLite persistence
    def _open_db(self, db_path: str) -> sqlite3.Connection:
        """
        Opens a SQLite database connection with safe defaults for concurrent access.
        """
        dirpath = os.path.dirname(db_path) or "."
        os.makedirs(dirpath, exist_ok=True)

        conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _ensure_schema(self) -> None:
        """
        Creates the required tables if they do not exist.
        """
        with self._db_lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cameras (
                    camera_id         TEXT PRIMARY KEY,
                    rtsp_link          TEXT NOT NULL,
                    threshold          REAL NOT NULL,
                    telegram_api_token TEXT NOT NULL,
                    telegram_chat_id   TEXT NOT NULL,
                    camera_nickname    TEXT NOT NULL,
                    cooldown_seconds   INTEGER NOT NULL,
                    created_at_unix    REAL NOT NULL
                );
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cameras_nickname ON cameras(camera_nickname);"
            )
            self._conn.commit()

    def _load_cameras_from_db(self) -> None:
        """
        Loads all cameras from SQLite into the in-memory registry.
        """
        with self._db_lock:
            cur = self._conn.execute(
                """
                SELECT camera_id, rtsp_link, threshold, telegram_api_token,
                       telegram_chat_id, camera_nickname, cooldown_seconds
                FROM cameras
                ORDER BY created_at_unix ASC;
                """
            )
            rows = cur.fetchall()

        with self._lock:
            self._cameras.clear()
            for r in rows:
                cfg = CameraConfig(
                    camera_id=str(r[0]),
                    rtsp_link=str(r[1]),
                    threshold=float(r[2]),
                    telegram_api_token=str(r[3]),
                    telegram_chat_id=str(r[4]),
                    camera_nickname=str(r[5]),
                    cooldown_seconds=int(r[6]),
                )
                self._cameras[cfg.camera_id] = cfg

    def _persist_camera(self, cfg: CameraConfig) -> None:
        """
        Inserts a new camera configuration into SQLite (atomic).
        """
        with self._db_lock:
            self._conn.execute(
                """
                INSERT INTO cameras (
                    camera_id, rtsp_link, threshold, telegram_api_token,
                    telegram_chat_id, camera_nickname, cooldown_seconds, created_at_unix
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    cfg.camera_id,
                    cfg.rtsp_link,
                    float(cfg.threshold),
                    cfg.telegram_api_token,
                    cfg.telegram_chat_id,
                    cfg.camera_nickname,
                    int(cfg.cooldown_seconds),
                    float(time.time()),
                ),
            )
            self._conn.commit()

    def _delete_camera_from_db(self, camera_id: str) -> None:
        """
        Deletes a camera configuration from SQLite (atomic).
        """
        with self._db_lock:
            self._conn.execute("DELETE FROM cameras WHERE camera_id = ?;", (camera_id,))
            self._conn.commit()

    # Camera management
    def add_camera(
        self,
        rtsp_link: str,
        camera_nickname: str,
        telegram_api_token: str,
        telegram_chat_id: str,
        threshold: float = 0.5,
        cooldown_seconds: int = 3600,
    ) -> str:
        """
        Adds a camera to the scheduler and persists it to SQLite.
        Safe under concurrent requests.
        """
        camera_id = uuid4().hex
        cfg = CameraConfig(
            camera_id=camera_id,
            rtsp_link=rtsp_link,
            threshold=float(threshold),
            telegram_api_token=telegram_api_token,
            telegram_chat_id=str(telegram_chat_id),
            camera_nickname=camera_nickname,
            cooldown_seconds=int(cooldown_seconds),
        )

        # Persist first (so it survives even if the process crashes right after)
        self._persist_camera(cfg)

        # Then register in-memory
        with self._lock:
            self._cameras[camera_id] = cfg

        return camera_id

    def remove_camera(self, camera_id: str) -> bool:
        """
        Removes a camera from the scheduler and from SQLite.
        """
        with self._lock:
            removed = self._cameras.pop(camera_id, None) is not None

        if removed:
            self._delete_camera_from_db(camera_id)
        return removed

    def get_status(self, camera_id: str) -> str:
        """
        Simple status check. If the camera exists in the registry, it is "ON".
        If the scheduler thread is not running, it is "OFF".
        """
        with self._lock:
            if camera_id not in self._cameras:
                return "NOT_FOUND"

        if not self.is_alive() or not self._running:
            return "OFF"
        return "ON"

    def list_cameras(self) -> List[CameraConfig]:
        """
        Returns a snapshot list of cameras currently registered in memory.
        """
        with self._lock:
            return list(self._cameras.values())

    def stop(self) -> None:
        """
        Stops the scheduler loop and closes the DB connection gracefully.
        """
        self._running = False
        try:
            with self._db_lock:
                self._conn.close()
        except Exception:
            pass

    # Thread main loop
    def run(self) -> None:
        """
        Round-robin loop across all registered cameras.
        """
        print("[INFO] CameraScheduler started.")
        while self._running:
            cameras_snapshot = self.list_cameras()
            if not cameras_snapshot:
                time.sleep(self.cycle_sleep)
                continue

            for cfg in cameras_snapshot:
                self._process_single_camera(cfg)

            time.sleep(self.cycle_sleep)

    # Per-camera processing
    def _process_single_camera(self, cfg: CameraConfig) -> None:
        # If there are no registered persons, skip detection work
        if not self.person_store.has_any_persons():
            return

        try:
            frame = self.face_embedder.capture_from_rtsp(cfg.rtsp_link)
        except Exception as exc:
            print(f"[WARN] Failed to capture from {cfg.camera_nickname}: {exc}")
            return

        detections = self.face_embedder.detect_and_embed(frame) or []
        if not detections:
            return

        for det in detections:
            emb = det["embedding"]
            bbox = det["bbox"]

            match: MatchResult = self.person_store.match_embedding(
                emb, threshold=cfg.threshold
            )
            if match.person_id is None:
                continue

            person_id = match.person_id
            now_ts = time.time()
            key = (cfg.camera_id, person_id)

            # Cooldown check (in-memory)
            with self._lock:
                last_ts = self._last_notified.get(key)
                if last_ts is not None and (now_ts - last_ts) < cfg.cooldown_seconds:
                    continue
                self._last_notified[key] = now_ts

            # Build caption with Brazil/São Paulo timezone
            brasilia_tz = pytz.timezone("America/Sao_Paulo")
            ts_brasilia = datetime.now(brasilia_tz)
            weekday = ts_brasilia.strftime("%A").capitalize()
            date_str = ts_brasilia.strftime("%d/%m/%Y")
            time_str = ts_brasilia.strftime("%H:%M:%S")

            caption = (
                "📸 Indivíduo detectado!\n"
                f"⌛ Dia da semana: {weekday}\n"
                f"🗓️ Data: {date_str}\n"
                f"⏰ Horário: {time_str}\n"
                f"🆔 Identificador: `{person_id}`\n"
                f"📷 Câmera: {cfg.camera_nickname}\n"
                f"🎯 Confiança da detecção: {(100 * match.score):.2f}\n"
            )

            out_path = f"/tmp/match_{cfg.camera_id}_{person_id}_{int(now_ts)}.jpg"
            self.face_embedder.draw_bbox_and_save(frame, bbox, out_path)
            TelegramClient.send_telegram_image(
                cfg.telegram_api_token,
                cfg.telegram_chat_id,
                out_path,
                caption,
            )

    def list_cameras_public(self) -> List[Dict[str, str | int | float]]:
        """
        Lists all persisted camera jobs from SQLite (source of truth).
        This is intended for API endpoints (similar to PersonStore.list_persons()).
        """
        with self._db_lock:
            cur = self._conn.execute(
                """
                SELECT camera_id, camera_nickname, rtsp_link, threshold,
                       cooldown_seconds, telegram_chat_id
                FROM cameras
                ORDER BY created_at_unix ASC;
                """
            )
            rows = cur.fetchall()

        return [
            {
                "camera_id": str(r[0]),
                "camera_nickname": str(r[1]),
                "rtsp_link": str(r[2]),
                "threshold": float(r[3]),
                "cooldown_seconds": int(r[4]),
                "telegram_chat_id": str(r[5]),
            }
            for r in rows
        ]

    def get_camera(self, camera_id: str) -> CameraConfig | None:
        """
        Returns a camera config from SQLite by id (source of truth).
        """
        with self._db_lock:
            cur = self._conn.execute(
                """
                SELECT camera_id, rtsp_link, threshold, telegram_api_token,
                    telegram_chat_id, camera_nickname, cooldown_seconds
                FROM cameras
                WHERE camera_id = ?;
                """,
                (camera_id,),
            )
            row = cur.fetchone()

        if row is None:
            return None

        return CameraConfig(
            camera_id=str(row[0]),
            rtsp_link=str(row[1]),
            threshold=float(row[2]),
            telegram_api_token=str(row[3]),
            telegram_chat_id=str(row[4]),
            camera_nickname=str(row[5]),
            cooldown_seconds=int(row[6]),
        )


    def has_any_cameras(self) -> bool:
        """
        Returns True if there is at least one persisted camera in SQLite.
        """
        with self._db_lock:
            cur = self._conn.execute("SELECT 1 FROM cameras LIMIT 1;")
            return cur.fetchone() is not None
