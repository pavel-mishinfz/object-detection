import unittest
from datetime import datetime
from unittest.mock import AsyncMock
from uuid import uuid4

from app.map.application.use_cases import (
    build_polygon,
    build_updated_polygon,
    create_polygon,
    delete_polygon,
    get_polygon,
    get_user_polygons,
    update_polygon,
)
from app.map.domain.errors import (
    PolygonAccessDeniedError,
    PolygonLimitExceededError,
    PolygonNameConflictError,
    PolygonNotFoundError,
    PolygonValidationError,
)
from app.map.domain.polygon import Coordinate, Polygon

VALID_COORDS = (
    (54.0, 44.0),
    (54.0, 44.5),
    (54.5, 44.5),
    (54.5, 44.0),
    (54.0, 44.0),
)


def _make_polygon(polygon_id=None, user_id=None, name="Полигон") -> Polygon:
    return Polygon(
        id=polygon_id or uuid4(),
        user_id=user_id or uuid4(),
        name=name,
        coordinates=tuple(Coordinate(lat=lat, lon=lon) for lat, lon in VALID_COORDS),
        created_at=datetime(2026, 1, 1),
    )


# ---------------------------------------------------------------------------
# build_polygon
# ---------------------------------------------------------------------------

class TestBuildPolygon(unittest.TestCase):

    def _build(self, **overrides):
        kwargs = dict(
            polygon_id=uuid4(),
            user_id=uuid4(),
            name="Тестовый полигон",
            coordinates=VALID_COORDS,
            created_at=datetime(2026, 1, 1),
        )
        kwargs.update(overrides)
        return build_polygon(**kwargs)

    def test_returns_polygon_with_correct_fields(self):
        polygon_id = uuid4()
        user_id = uuid4()
        created_at = datetime(2026, 1, 1)
        result = self._build(polygon_id=polygon_id, user_id=user_id,
                             name="Тест", created_at=created_at)
        self.assertIsInstance(result, Polygon)
        self.assertEqual(result.id, polygon_id)
        self.assertEqual(result.user_id, user_id)
        self.assertEqual(result.name, "Тест")
        self.assertEqual(result.created_at, created_at)
        self.assertEqual(len(result.coordinates), len(VALID_COORDS))

    def test_raises_if_polygon_not_closed(self):
        with self.assertRaises(PolygonValidationError):
            self._build(coordinates=((54.0, 44.0), (54.0, 44.5), (54.5, 44.5)))

    def test_raises_if_fewer_than_three_unique_coordinates(self):
        with self.assertRaises(PolygonValidationError):
            self._build(coordinates=((54.0, 44.0), (54.5, 44.5), (54.0, 44.0)))

    def test_raises_if_duplicate_coordinates_give_less_than_three_unique(self):
        with self.assertRaises(PolygonValidationError):
            self._build(coordinates=((54.0, 44.0),) * 4)

    def test_raises_if_name_is_empty(self):
        with self.assertRaises(PolygonValidationError):
            self._build(name="")

    def test_raises_if_name_is_blank(self):
        with self.assertRaises(PolygonValidationError):
            self._build(name="   ")

    def test_raises_if_lat_exceeds_90(self):
        coords = ((91.0, 44.0), (54.0, 44.5), (54.5, 44.5), (54.5, 44.0), (91.0, 44.0))
        with self.assertRaises(PolygonValidationError):
            self._build(coordinates=coords)

    def test_raises_if_lat_below_minus_90(self):
        coords = ((-91.0, 44.0), (54.0, 44.5), (54.5, 44.5), (54.5, 44.0), (-91.0, 44.0))
        with self.assertRaises(PolygonValidationError):
            self._build(coordinates=coords)

    def test_raises_if_lon_exceeds_180(self):
        coords = ((54.0, 181.0), (54.0, 44.5), (54.5, 44.5), (54.5, 44.0), (54.0, 181.0))
        with self.assertRaises(PolygonValidationError):
            self._build(coordinates=coords)

    def test_raises_if_lon_below_minus_180(self):
        coords = ((54.0, -181.0), (54.0, 44.5), (54.5, 44.5), (54.5, 44.0), (54.0, -181.0))
        with self.assertRaises(PolygonValidationError):
            self._build(coordinates=coords)

    def test_raises_if_polygon_self_intersects(self):
        bowtie = ((54.0, 44.0), (54.5, 44.5), (54.0, 44.5), (54.5, 44.0), (54.0, 44.0))
        with self.assertRaises(PolygonValidationError):
            self._build(coordinates=bowtie)

    def test_raises_if_area_too_small(self):
        tiny = (
            (54.0000, 44.0000), (54.0000, 44.0001),
            (54.0001, 44.0001), (54.0001, 44.0000), (54.0000, 44.0000),
        )
        with self.assertRaises(PolygonValidationError):
            self._build(coordinates=tiny)

    def test_raises_if_area_too_large(self):
        huge = ((50.0, 40.0), (50.0, 50.0), (60.0, 50.0), (60.0, 40.0), (50.0, 40.0))
        with self.assertRaises(PolygonValidationError):
            self._build(coordinates=huge)


# ---------------------------------------------------------------------------
# build_updated_polygon
# ---------------------------------------------------------------------------

class TestBuildUpdatedPolygon(unittest.TestCase):

    def test_returns_polygon_with_updated_name(self):
        existing = _make_polygon(name="Старое")
        result = build_updated_polygon(existing, name="Новое", coordinates=VALID_COORDS)
        self.assertEqual(result.name, "Новое")

    def test_preserves_id_user_id_and_created_at(self):
        existing = _make_polygon()
        result = build_updated_polygon(existing, name="Новое", coordinates=VALID_COORDS)
        self.assertEqual(result.id, existing.id)
        self.assertEqual(result.user_id, existing.user_id)
        self.assertEqual(result.created_at, existing.created_at)

    def test_raises_for_empty_name(self):
        with self.assertRaises(PolygonValidationError):
            build_updated_polygon(_make_polygon(), name="", coordinates=VALID_COORDS)

    def test_raises_for_invalid_geometry(self):
        open_coords = ((54.0, 44.0), (54.0, 44.5), (54.5, 44.5))
        with self.assertRaises(PolygonValidationError):
            build_updated_polygon(_make_polygon(), name="X", coordinates=open_coords)


# ---------------------------------------------------------------------------
# create_polygon
# ---------------------------------------------------------------------------

class TestCreatePolygon(unittest.IsolatedAsyncioTestCase):

    def _make_repo(self, *, count: int = 0, name_exists: bool = False) -> AsyncMock:
        repo = AsyncMock()
        repo.count_by_user.return_value = count
        repo.exists_with_name.return_value = name_exists
        return repo

    async def test_calls_repo_save_on_success(self):
        repo = self._make_repo()
        await create_polygon(
            polygon_id=uuid4(), user_id=uuid4(),
            name="Новый полигон", coordinates=VALID_COORDS, repo=repo,
        )
        repo.save.assert_called_once()

    async def test_saved_polygon_has_correct_id_and_user(self):
        repo = self._make_repo()
        polygon_id, user_id = uuid4(), uuid4()
        await create_polygon(
            polygon_id=polygon_id, user_id=user_id,
            name="Полигон", coordinates=VALID_COORDS, repo=repo,
        )
        saved: Polygon = repo.save.call_args[0][0]
        self.assertEqual(saved.id, polygon_id)
        self.assertEqual(saved.user_id, user_id)

    async def test_raises_when_user_polygon_limit_reached(self):
        repo = self._make_repo(count=100)
        with self.assertRaises(PolygonLimitExceededError):
            await create_polygon(
                polygon_id=uuid4(), user_id=uuid4(),
                name="Полигон", coordinates=VALID_COORDS, repo=repo,
            )

    async def test_does_not_save_when_limit_exceeded(self):
        repo = self._make_repo(count=100)
        with self.assertRaises(PolygonLimitExceededError):
            await create_polygon(
                polygon_id=uuid4(), user_id=uuid4(),
                name="Полигон", coordinates=VALID_COORDS, repo=repo,
            )
        repo.save.assert_not_called()

    async def test_raises_when_name_already_exists_for_user(self):
        repo = self._make_repo(name_exists=True)
        with self.assertRaises(PolygonNameConflictError):
            await create_polygon(
                polygon_id=uuid4(), user_id=uuid4(),
                name="Существующий", coordinates=VALID_COORDS, repo=repo,
            )

    async def test_does_not_save_when_name_conflict(self):
        repo = self._make_repo(name_exists=True)
        with self.assertRaises(PolygonNameConflictError):
            await create_polygon(
                polygon_id=uuid4(), user_id=uuid4(),
                name="Существующий", coordinates=VALID_COORDS, repo=repo,
            )
        repo.save.assert_not_called()

    async def test_limit_check_uses_correct_user_id(self):
        repo = self._make_repo()
        user_id = uuid4()
        await create_polygon(
            polygon_id=uuid4(), user_id=user_id,
            name="Полигон", coordinates=VALID_COORDS, repo=repo,
        )
        repo.count_by_user.assert_called_once_with(user_id)

    async def test_name_check_uses_correct_user_and_name(self):
        repo = self._make_repo()
        user_id = uuid4()
        await create_polygon(
            polygon_id=uuid4(), user_id=user_id,
            name="Уникальное имя", coordinates=VALID_COORDS, repo=repo,
        )
        repo.exists_with_name.assert_called_once_with(user_id, "Уникальное имя")


# ---------------------------------------------------------------------------
# update_polygon
# ---------------------------------------------------------------------------

class TestUpdatePolygon(unittest.IsolatedAsyncioTestCase):

    def _make_repo(self, *, polygon=None, name_exists: bool = False) -> AsyncMock:
        repo = AsyncMock()
        repo.find_by_id.return_value = polygon
        repo.exists_with_name.return_value = name_exists
        return repo

    async def test_calls_repo_update_on_success(self):
        user_id = uuid4()
        existing = _make_polygon(user_id=user_id)
        repo = self._make_repo(polygon=existing)
        await update_polygon(
            polygon_id=existing.id, user_id=user_id,
            name="Новое имя", coordinates=VALID_COORDS, repo=repo,
        )
        repo.update.assert_called_once()

    async def test_updated_polygon_has_new_name(self):
        user_id = uuid4()
        existing = _make_polygon(user_id=user_id)
        repo = self._make_repo(polygon=existing)
        await update_polygon(
            polygon_id=existing.id, user_id=user_id,
            name="Новое имя", coordinates=VALID_COORDS, repo=repo,
        )
        updated: Polygon = repo.update.call_args[0][0]
        self.assertEqual(updated.name, "Новое имя")

    async def test_updated_polygon_preserves_id_user_id_created_at(self):
        user_id = uuid4()
        existing = _make_polygon(user_id=user_id)
        repo = self._make_repo(polygon=existing)
        await update_polygon(
            polygon_id=existing.id, user_id=user_id,
            name="Новое", coordinates=VALID_COORDS, repo=repo,
        )
        updated: Polygon = repo.update.call_args[0][0]
        self.assertEqual(updated.id, existing.id)
        self.assertEqual(updated.user_id, existing.user_id)
        self.assertEqual(updated.created_at, existing.created_at)

    async def test_raises_if_polygon_not_found(self):
        repo = self._make_repo(polygon=None)
        with self.assertRaises(PolygonNotFoundError):
            await update_polygon(
                polygon_id=uuid4(), user_id=uuid4(),
                name="X", coordinates=VALID_COORDS, repo=repo,
            )

    async def test_does_not_update_when_not_found(self):
        repo = self._make_repo(polygon=None)
        with self.assertRaises(PolygonNotFoundError):
            await update_polygon(
                polygon_id=uuid4(), user_id=uuid4(),
                name="X", coordinates=VALID_COORDS, repo=repo,
            )
        repo.update.assert_not_called()

    async def test_raises_if_not_owner(self):
        existing = _make_polygon(user_id=uuid4())
        repo = self._make_repo(polygon=existing)
        with self.assertRaises(PolygonAccessDeniedError):
            await update_polygon(
                polygon_id=existing.id, user_id=uuid4(),
                name="X", coordinates=VALID_COORDS, repo=repo,
            )

    async def test_does_not_update_when_not_owner(self):
        existing = _make_polygon(user_id=uuid4())
        repo = self._make_repo(polygon=existing)
        with self.assertRaises(PolygonAccessDeniedError):
            await update_polygon(
                polygon_id=existing.id, user_id=uuid4(),
                name="X", coordinates=VALID_COORDS, repo=repo,
            )
        repo.update.assert_not_called()

    async def test_raises_if_name_taken_by_another_polygon(self):
        user_id = uuid4()
        existing = _make_polygon(user_id=user_id)
        repo = self._make_repo(polygon=existing, name_exists=True)
        with self.assertRaises(PolygonNameConflictError):
            await update_polygon(
                polygon_id=existing.id, user_id=user_id,
                name="Занятое", coordinates=VALID_COORDS, repo=repo,
            )

    async def test_does_not_update_when_name_conflict(self):
        user_id = uuid4()
        existing = _make_polygon(user_id=user_id)
        repo = self._make_repo(polygon=existing, name_exists=True)
        with self.assertRaises(PolygonNameConflictError):
            await update_polygon(
                polygon_id=existing.id, user_id=user_id,
                name="Занятое", coordinates=VALID_COORDS, repo=repo,
            )
        repo.update.assert_not_called()

    async def test_name_check_excludes_current_polygon(self):
        user_id = uuid4()
        existing = _make_polygon(user_id=user_id)
        repo = self._make_repo(polygon=existing)
        await update_polygon(
            polygon_id=existing.id, user_id=user_id,
            name="Любое", coordinates=VALID_COORDS, repo=repo,
        )
        repo.exists_with_name.assert_called_once_with(user_id, "Любое", existing.id)


# ---------------------------------------------------------------------------
# get_polygon
# ---------------------------------------------------------------------------

class TestGetPolygon(unittest.IsolatedAsyncioTestCase):

    def _make_repo(self, *, polygon=None) -> AsyncMock:
        repo = AsyncMock()
        repo.find_by_id.return_value = polygon
        return repo

    async def test_returns_polygon_on_success(self):
        user_id = uuid4()
        existing = _make_polygon(user_id=user_id)
        repo = self._make_repo(polygon=existing)
        result = await get_polygon(
            polygon_id=existing.id, user_id=user_id, repo=repo,
        )
        self.assertEqual(result, existing)

    async def test_raises_if_polygon_not_found(self):
        repo = self._make_repo(polygon=None)
        with self.assertRaises(PolygonNotFoundError):
            await get_polygon(polygon_id=uuid4(), user_id=uuid4(), repo=repo)

    async def test_raises_if_not_owner(self):
        existing = _make_polygon(user_id=uuid4())
        repo = self._make_repo(polygon=existing)
        with self.assertRaises(PolygonAccessDeniedError):
            await get_polygon(polygon_id=existing.id, user_id=uuid4(), repo=repo)

    async def test_uses_correct_polygon_id(self):
        user_id = uuid4()
        existing = _make_polygon(user_id=user_id)
        repo = self._make_repo(polygon=existing)
        await get_polygon(polygon_id=existing.id, user_id=user_id, repo=repo)
        repo.find_by_id.assert_called_once_with(existing.id)


# ---------------------------------------------------------------------------
# delete_polygon
# ---------------------------------------------------------------------------

class TestDeletePolygon(unittest.IsolatedAsyncioTestCase):

    def _make_repo(self, *, polygon=None) -> AsyncMock:
        repo = AsyncMock()
        repo.find_by_id.return_value = polygon
        return repo

    async def test_calls_repo_delete_on_success(self):
        user_id = uuid4()
        existing = _make_polygon(user_id=user_id)
        repo = self._make_repo(polygon=existing)
        await delete_polygon(polygon_id=existing.id, user_id=user_id, repo=repo)
        repo.delete.assert_called_once()

    async def test_uses_correct_polygon_id_for_delete(self):
        user_id = uuid4()
        existing = _make_polygon(user_id=user_id)
        repo = self._make_repo(polygon=existing)
        await delete_polygon(polygon_id=existing.id, user_id=user_id, repo=repo)
        repo.delete.assert_called_once_with(existing.id)

    async def test_raises_if_polygon_not_found(self):
        repo = self._make_repo(polygon=None)
        with self.assertRaises(PolygonNotFoundError):
            await delete_polygon(polygon_id=uuid4(), user_id=uuid4(), repo=repo)

    async def test_does_not_delete_when_not_found(self):
        repo = self._make_repo(polygon=None)
        with self.assertRaises(PolygonNotFoundError):
            await delete_polygon(polygon_id=uuid4(), user_id=uuid4(), repo=repo)
        repo.delete.assert_not_called()

    async def test_raises_if_not_owner(self):
        existing = _make_polygon(user_id=uuid4())
        repo = self._make_repo(polygon=existing)
        with self.assertRaises(PolygonAccessDeniedError):
            await delete_polygon(polygon_id=existing.id, user_id=uuid4(), repo=repo)

    async def test_does_not_delete_when_not_owner(self):
        existing = _make_polygon(user_id=uuid4())
        repo = self._make_repo(polygon=existing)
        with self.assertRaises(PolygonAccessDeniedError):
            await delete_polygon(polygon_id=existing.id, user_id=uuid4(), repo=repo)
        repo.delete.assert_not_called()


# ---------------------------------------------------------------------------
# get_user_polygons
# ---------------------------------------------------------------------------

class TestGetUserPolygons(unittest.IsolatedAsyncioTestCase):

    def _make_repo(self, *, polygons=None) -> AsyncMock:
        repo = AsyncMock()
        repo.find_by_user.return_value = polygons if polygons is not None else []
        return repo

    async def test_returns_list_of_user_polygons(self):
        user_id = uuid4()
        polygons = [_make_polygon(user_id=user_id), _make_polygon(user_id=user_id)]
        repo = self._make_repo(polygons=polygons)
        result = await get_user_polygons(user_id=user_id, repo=repo)
        self.assertEqual(result, polygons)

    async def test_returns_empty_list_when_user_has_no_polygons(self):
        repo = self._make_repo(polygons=[])
        result = await get_user_polygons(user_id=uuid4(), repo=repo)
        self.assertEqual(result, [])

    async def test_uses_correct_user_id(self):
        user_id = uuid4()
        repo = self._make_repo()
        await get_user_polygons(user_id=user_id, repo=repo)
        repo.find_by_user.assert_called_once_with(user_id)


if __name__ == "__main__":
    unittest.main()
