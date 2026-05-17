from collections import defaultdict
from collections.abc import Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

type Handler = Callable[[object, AsyncSession], Awaitable[list[Callable] | None]]


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[type, list[Handler]] = defaultdict(list)

    def subscribe(self, event_type: type, handler: Handler) -> None:
        self._handlers[event_type].append(handler)

    async def publish(self, event: object, session: AsyncSession) -> list[Callable]:
        callbacks: list[Callable] = []
        for handler in self._handlers.get(type(event), []):
            result = await handler(event, session)
            if result:
                callbacks.extend(result)
        return callbacks
