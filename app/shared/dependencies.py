from collections.abc import Callable

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.shared.contracts import IEventPublisher
from app.shared.database import get_session
from app.shared.event_bus import EventBus


class SessionEventPublisher:
    def __init__(self, bus: EventBus, session: AsyncSession) -> None:
        self._bus = bus
        self._session = session
        self._post_commit: list[Callable] = []

    async def publish(self, event: object) -> None:
        callbacks = await self._bus.publish(event, self._session)
        self._post_commit.extend(callbacks)

    async def run_post_commit(self) -> None:
        for cb in self._post_commit:
            await cb()


def get_event_publisher(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> IEventPublisher:
    bus: EventBus = request.app.state.event_bus
    return SessionEventPublisher(bus, session)
