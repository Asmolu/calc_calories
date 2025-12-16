from typing import Optional

from celery import Celery

celery_app: Optional[Celery] = None


def init_celery(broker_url: str | None = None) -> Celery:
    """Create and return a Celery application instance."""

    global celery_app
    celery_app = Celery("calai", broker=broker_url or "redis://localhost:6379/0")
    celery_app.conf.task_default_queue = "default"
    return celery_app