from collections import defaultdict
from collections.abc import Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

type Handler = Callable[[object, AsyncSession], Awaitable[list[Callable] | None]]

_handlers: dict[type, list[Handler]] = defaultdict(list)


def subscribe(event_type: type, handler: Handler) -> None:
    _handlers[event_type].append(handler)


async def publish(event: object, session: AsyncSession) -> list[Callable]:
    callbacks: list[Callable] = []
    for handler in _handlers.get(type(event), []):
        result = await handler(event, session)
        if result:
            callbacks.extend(result)
    return callbacks
