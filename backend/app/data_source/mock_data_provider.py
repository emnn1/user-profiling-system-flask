"""模拟数据源与实时 API 接口。

该模块提供：
- 基于 Faker 的离线数据批量生成方法；
- 模拟历史事件流的构造方法；
- 一个异步的 `MockRealtimeAPI`，用于按需拉取用户信息与实时事件。

所有后端组件应通过此模块提供的接口访问模拟数据，以便未来平滑衔接真实数据中台。
"""
from __future__ import annotations

import asyncio
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, Iterable, Literal, Sequence

import pandas as pd
from faker import Faker

DEFAULT_LOCALE = "zh_CN"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "generated_data"
DEFAULT_FILE_FORMAT: Literal["csv", "parquet"] = "csv"
DEFAULT_USER_COUNT = 1_000
DEFAULT_PRODUCT_COUNT = 25
DEFAULT_APP_COUNT = 30
DEFAULT_AVG_EVENTS_PER_USER = 20

PLAN_TYPES = ["基础套餐", "5G畅享套餐", "家庭融合套餐", "校园流量套餐"]
USER_LEVELS = ["普通用户", "银卡", "金卡", "白金"]
DEVICE_BRANDS = ["华为", "小米", "苹果", "OPPO", "vivo", "荣耀"]
APP_NAMES = [
    "微信",
    "抖音",
    "淘宝",
    "支付宝",
    "QQ音乐",
    "网易云音乐",
    "钉钉",
    "京东",
    "哔哩哔哩",
    "美团",
]
PRODUCT_TEMPLATES = [
    ("5G畅享套餐", 129.0),
    ("5G至尊套餐", 199.0),
    ("家庭宽带融合", 159.0),
    ("校园流量包", 69.0),
    ("国际漫游包", 299.0),
]


class EventType(str, Enum):
    """用户行为事件类型。"""

    CALL = "通话"
    ORDER = "订购"
    APP_USAGE = "APP使用"
    CLICK = "点击"


@dataclass(slots=True)
class User:
    """用户基础画像信息。"""

    user_id: str
    plan_type: str
    monthly_fee: float
    user_level: str
    tenure_months: int
    device_brand: str


@dataclass(slots=True)
class Product:
    """产品信息定义。"""

    product_id: str
    product_name: str
    price: float


@dataclass(slots=True)
class App:
    """移动应用信息。"""

    app_id: str
    app_name: str


@dataclass(slots=True)
class Event:
    """用户行为事件。"""

    event_id: str
    timestamp: datetime
    user_id: str
    event_type: EventType
    target_user_id: str | None = None
    product_id: str | None = None
    app_id: str | None = None
    duration_seconds: int | None = None


def generate_initial_data(
    num_users: int,
    num_products: int,
    num_apps: int,
    *,
    output_dir: str | Path | None = None,
    file_format: Literal["csv", "parquet"] = DEFAULT_FILE_FORMAT,
) -> tuple[list[User], list[Product], list[App]]:
    """批量生成用户、产品与应用的静态模拟数据并持久化。"""

    if num_users <= 0 or num_products <= 0 or num_apps <= 0:
        raise ValueError("num_users, num_products, num_apps 必须为正整数")

    faker = Faker(DEFAULT_LOCALE)
    output_path = Path(output_dir) if output_dir else DEFAULT_DATA_DIR
    _ensure_directory(output_path)

    users = [
        User(
            user_id=faker.uuid4(),
            plan_type=random.choice(PLAN_TYPES),
            monthly_fee=round(random.uniform(39, 399), 2),
            user_level=random.choice(USER_LEVELS),
            tenure_months=random.randint(1, 180),
            device_brand=random.choice(DEVICE_BRANDS),
        )
        for _ in range(num_users)
    ]

    products = [
        Product(
            product_id=faker.uuid4(),
            product_name=name if idx < len(PRODUCT_TEMPLATES) else faker.catch_phrase(),
            price=round(price if idx < len(PRODUCT_TEMPLATES) else random.uniform(49, 499), 2),
        )
        for idx, (name, price) in enumerate(random.choices(PRODUCT_TEMPLATES, k=num_products))
    ]

    apps = [
        App(
            app_id=faker.uuid4(),
            app_name=APP_NAMES[idx] if idx < len(APP_NAMES) else faker.company(),
        )
        for idx in range(num_apps)
    ]

    _save_dataframe(pd.DataFrame([asdict(u) for u in users]), output_path, "users", file_format)
    _save_dataframe(pd.DataFrame([asdict(p) for p in products]), output_path, "products", file_format)
    _save_dataframe(pd.DataFrame([asdict(a) for a in apps]), output_path, "apps", file_format)

    return users, products, apps


def generate_historical_events(
    users: Iterable[User],
    products: Iterable[Product],
    apps: Iterable[App],
    start_date: datetime,
    end_date: datetime,
    *,
    output_dir: str | Path | None = None,
    file_format: Literal["csv", "parquet"] = DEFAULT_FILE_FORMAT,
    avg_events_per_user: int = DEFAULT_AVG_EVENTS_PER_USER,
) -> list[Event]:
    """为给定用户集合生成历史事件序列并持久化。"""

    if start_date >= end_date:
        raise ValueError("start_date 必须早于 end_date")
    if avg_events_per_user <= 0:
        raise ValueError("avg_events_per_user 必须为正整数")

    faker = Faker(DEFAULT_LOCALE)
    output_path = Path(output_dir) if output_dir else DEFAULT_DATA_DIR
    _ensure_directory(output_path)

    user_list = _ensure_sequence(users, _row_to_user)
    product_list = _ensure_sequence(products, _row_to_product)
    app_list = _ensure_sequence(apps, _row_to_app)

    events: list[Event] = []
    for user in user_list:
        event_count = max(1, int(random.gauss(avg_events_per_user, avg_events_per_user // 3 or 1)))
        for _ in range(event_count):
            event_type = random.choice(list(EventType))
            timestamp = faker.date_time_between_dates(start_date=start_date, end_date=end_date)
            duration: int | None = None
            target_user_id: str | None = None
            product_id: str | None = None
            app_id: str | None = None

            if event_type is EventType.CALL:
                other_user = random.choice(user_list)
                target_user_id = other_user.user_id if other_user.user_id != user.user_id else None
                duration = random.randint(30, 1800)
            elif event_type is EventType.ORDER:
                product_id = random.choice(product_list).product_id
            elif event_type is EventType.APP_USAGE:
                app_id = random.choice(app_list).app_id
                duration = random.randint(60, 3600)
            elif event_type is EventType.CLICK:
                if random.random() < 0.5:
                    product_id = random.choice(product_list).product_id
                else:
                    app_id = random.choice(app_list).app_id

            events.append(
                Event(
                    event_id=faker.uuid4(),
                    timestamp=timestamp,
                    user_id=user.user_id,
                    event_type=event_type,
                    target_user_id=target_user_id,
                    product_id=product_id,
                    app_id=app_id,
                    duration_seconds=duration,
                )
            )

    events.sort(key=lambda event: event.timestamp)
    _save_dataframe(
        pd.DataFrame([_event_to_record(event) for event in events]),
        output_path,
        "events",
        file_format,
    )
    return events


class MockRealtimeAPI:
    """模拟实时数据接口，兼容未来真实数据服务的契约。"""

    def __init__(
        self,
        *,
        data_dir: str | Path | None = None,
        file_format: Literal["csv", "parquet"] = DEFAULT_FILE_FORMAT,
        realtime_delay_seconds: float = 1.0,
    ) -> None:
        """构造模拟实时 API。

        参数说明：

        - ``data_dir``: 生成与存放模拟数据集的目录；
        - ``file_format``: 底层数据格式，需与离线生成逻辑保持一致；
        - ``realtime_delay_seconds``: 通过 ``asyncio.sleep`` 控制事件推送速度，便于模拟准实时场景。
        """
        self._faker = Faker(DEFAULT_LOCALE)
        self._data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self._file_format = file_format
        self._realtime_delay = max(0.0, realtime_delay_seconds)

        self._ensure_seed_data()

        self._users_df = _load_dataframe(self._data_dir, "users", self._file_format)
        self._products_df = _load_dataframe(self._data_dir, "products", self._file_format)
        self._apps_df = _load_dataframe(self._data_dir, "apps", self._file_format)
        self._events_df = _load_dataframe(self._data_dir, "events", self._file_format).sort_values("timestamp")

        for column in ("timestamp",):
            if column in self._events_df.columns:
                self._events_df[column] = pd.to_datetime(self._events_df[column])

        self._users: list[User] = [_row_to_user(row) for _, row in self._users_df.iterrows()]
        self._products: list[Product] = [_row_to_product(row) for _, row in self._products_df.iterrows()]
        self._apps: list[App] = [_row_to_app(row) for _, row in self._apps_df.iterrows()]
    
    @property
    def data_dir(self) -> Path:
        """返回当前使用的数据目录，供其他模块探测持久化文件。"""
        return self._data_dir

    @property
    def file_format(self) -> Literal["csv", "parquet"]:
        """返回底层数据集的文件格式，确保数据读写保持兼容。"""
        return self._file_format

    async def get_new_events(self, since_timestamp: datetime) -> AsyncIterator[Event]:
        """模拟从 ``since_timestamp`` 之后持续产生事件。

        迭代逻辑：优先消费离线事件表中尚未读取的事件；当耗尽后，
        即时构造一条新事件并写回缓存，以保证调用方持续获得数据。
        """

        last_timestamp = since_timestamp
        cursor = self._find_event_index(last_timestamp)

        while True:
            if cursor < len(self._events_df):
                row = self._events_df.iloc[cursor]
                event = _row_to_event(row)
                cursor += 1
            else:
                event = self._generate_live_event(last_timestamp)
                self._append_event(event)
                cursor = len(self._events_df)

            last_timestamp = max(last_timestamp, event.timestamp)

            if self._realtime_delay:
                await asyncio.sleep(self._realtime_delay)
            yield event

    async def get_user_batch(self, batch_size: int, page: int = 1) -> list[User]:
        """按分页拉取用户画像数据，供摄取服务进行缓存预热。"""

        if batch_size <= 0 or page <= 0:
            raise ValueError("batch_size 与 page 必须为正整数")

        start = batch_size * (page - 1)
        end = start + batch_size
        if start >= len(self._users_df):
            return []

        batch = self._users_df.iloc[start:end]
        return [_row_to_user(row) for _, row in batch.iterrows()]

    def _ensure_seed_data(self) -> None:
        """确保基础数据集存在，不足时触发离线生成流程。"""
        if not self._data_dir.exists():
            _ensure_directory(self._data_dir)

        users_path = _dataset_path(self._data_dir, "users", self._file_format)
        products_path = _dataset_path(self._data_dir, "products", self._file_format)
        apps_path = _dataset_path(self._data_dir, "apps", self._file_format)
        events_path = _dataset_path(self._data_dir, "events", self._file_format)

        need_initial = not (users_path.exists() and products_path.exists() and apps_path.exists())
        if need_initial:
            users, products, apps = generate_initial_data(
                DEFAULT_USER_COUNT,
                DEFAULT_PRODUCT_COUNT,
                DEFAULT_APP_COUNT,
                output_dir=self._data_dir,
                file_format=self._file_format,
            )
        else:
            users = products = apps = None

        if not events_path.exists():
            if users is None or products is None or apps is None:
                users = [_row_to_user(row) for _, row in _load_dataframe(self._data_dir, "users", self._file_format).iterrows()]
                products = [_row_to_product(row) for _, row in _load_dataframe(self._data_dir, "products", self._file_format).iterrows()]
                apps = [_row_to_app(row) for _, row in _load_dataframe(self._data_dir, "apps", self._file_format).iterrows()]

            now = datetime.utcnow()
            generate_historical_events(
                users,
                products,
                apps,
                start_date=now - timedelta(days=30),
                end_date=now,
                output_dir=self._data_dir,
                file_format=self._file_format,
            )

    def _find_event_index(self, since_timestamp: datetime) -> int:
        """返回事件表中首个晚于 ``since_timestamp`` 的行索引。"""
        newer_mask = self._events_df["timestamp"] > pd.Timestamp(since_timestamp)
        indices = newer_mask[newer_mask].index
        return int(indices[0]) if len(indices) else len(self._events_df)

    def _generate_live_event(self, baseline: datetime) -> Event:
        """根据基准时间生成一条新的实时事件，并保证字段合法。"""
        user = random.choice(self._users)
        event_type = random.choice(list(EventType))
        timestamp = baseline + timedelta(seconds=random.randint(1, 30))

        target_user_id: str | None = None
        product_id: str | None = None
        app_id: str | None = None
        duration_seconds: int | None = None

        if event_type is EventType.CALL:
            target_user = random.choice(self._users)
            target_user_id = target_user.user_id if target_user.user_id != user.user_id else None
            duration_seconds = random.randint(30, 900)
        elif event_type is EventType.ORDER:
            product_id = random.choice(self._products).product_id
        elif event_type is EventType.APP_USAGE:
            app_id = random.choice(self._apps).app_id
            duration_seconds = random.randint(120, 1800)
        elif event_type is EventType.CLICK:
            if random.random() < 0.5:
                product_id = random.choice(self._products).product_id
            else:
                app_id = random.choice(self._apps).app_id

        return Event(
            event_id=self._faker.uuid4(),
            timestamp=timestamp,
            user_id=user.user_id,
            event_type=event_type,
            target_user_id=target_user_id,
            product_id=product_id,
            app_id=app_id,
            duration_seconds=duration_seconds,
        )

    def _append_event(self, event: Event) -> None:
        """将实时事件附加到数据框中，维持时间序有序。"""
        record = _event_to_record(event)
        record_df = pd.DataFrame([record])
        record_df["timestamp"] = pd.to_datetime(record_df["timestamp"])
        self._events_df = pd.concat([self._events_df, record_df], ignore_index=True)


def _ensure_directory(path: Path) -> None:
    """创建目录并忽略已存在场景，供数据生成与持久化使用。"""
    path.mkdir(parents=True, exist_ok=True)


def _dataset_path(data_dir: Path, name: str, file_format: Literal["csv", "parquet"]) -> Path:
    """根据数据集名称拼接文件路径。"""
    return data_dir / f"{name}.{file_format}"


def _save_dataframe(
    df: pd.DataFrame,
    data_dir: Path,
    name: str,
    file_format: Literal["csv", "parquet"],
) -> Path:
    """将 ``DataFrame`` 序列化到磁盘，并处理时间戳格式。"""
    path = _dataset_path(data_dir, name, file_format)
    serializable_df = df.copy()
    for column in serializable_df.columns:
        if pd.api.types.is_datetime64_any_dtype(serializable_df[column]):
            serializable_df[column] = serializable_df[column].dt.strftime("%Y-%m-%dT%H:%M:%S")

    if file_format == "csv":
        serializable_df.to_csv(path, index=False)
    elif file_format == "parquet":
        try:
            serializable_df.to_parquet(path, index=False)
        except ImportError as exc:  # pragma: no cover - 依赖缺失时暴露
            raise RuntimeError("保存为 Parquet 需要安装 pyarrow 或 fastparquet") from exc
    else:  # pragma: no cover - 理论上不会触发
        raise ValueError("不支持的文件格式")
    return path


def _load_dataframe(
    data_dir: Path,
    name: str,
    file_format: Literal["csv", "parquet"],
) -> pd.DataFrame:
    """读取指定名称的数据集并返回 ``DataFrame``。"""
    path = _dataset_path(data_dir, name, file_format)
    if not path.exists():
        raise FileNotFoundError(f"未找到数据文件: {path}")

    if file_format == "csv":
        df = pd.read_csv(path)
    elif file_format == "parquet":
        df = pd.read_parquet(path)
    else:  # pragma: no cover
        raise ValueError("不支持的文件格式")
    return df


def _event_to_record(event: Event) -> dict[str, object]:
    """将事件对象转换为便于持久化的字典结构。"""
    record = asdict(event)
    record["event_type"] = event.event_type.value
    record["timestamp"] = event.timestamp.strftime("%Y-%m-%dT%H:%M:%S")
    return record


def _row_to_user(row: pd.Series) -> User:
    """根据数据行恢复 :class:`User` dataclass。"""
    return User(
        user_id=str(row["user_id"]),
        plan_type=str(row["plan_type"]),
        monthly_fee=float(row["monthly_fee"]),
        user_level=str(row["user_level"]),
        tenure_months=int(row["tenure_months"]),
        device_brand=str(row["device_brand"]),
    )


def _row_to_product(row: pd.Series) -> Product:
    """根据数据行恢复 :class:`Product` dataclass。"""
    return Product(
        product_id=str(row["product_id"]),
        product_name=str(row["product_name"]),
        price=float(row["price"]),
    )


def _row_to_app(row: pd.Series) -> App:
    """根据数据行恢复 :class:`App` dataclass。"""
    return App(
        app_id=str(row["app_id"]),
        app_name=str(row["app_name"]),
    )


def _row_to_event(row: pd.Series) -> Event:
    """根据数据行恢复 :class:`Event` dataclass，自动转换时间戳。"""
    timestamp = row["timestamp"]
    if not isinstance(timestamp, datetime):
        timestamp = pd.to_datetime(timestamp).to_pydatetime()

    return Event(
        event_id=str(row["event_id"]),
        timestamp=timestamp,
        user_id=str(row["user_id"]),
        event_type=EventType(str(row["event_type"])),
        target_user_id=row.get("target_user_id") if pd.notna(row.get("target_user_id")) else None,
        product_id=row.get("product_id") if pd.notna(row.get("product_id")) else None,
        app_id=row.get("app_id") if pd.notna(row.get("app_id")) else None,
        duration_seconds=int(row["duration_seconds"]) if pd.notna(row.get("duration_seconds")) else None,
    )


def _ensure_sequence(
    items: Iterable[object],
    converter: callable,
) -> list:
    """保证输入序列统一转换为 dataclass 列表。"""
    result = []
    for item in items:
        if isinstance(item, (User, Product, App)):
            result.append(item)
        elif isinstance(item, pd.Series):
            result.append(converter(item))
        elif isinstance(item, dict):
            result.append(converter(pd.Series(item)))
        else:
            raise TypeError(f"无法识别的记录类型: {type(item)!r}")
    return result


__all__ = [
    "App",
    "Event",
    "EventType",
    "MockRealtimeAPI",
    "Product",
    "User",
    "generate_historical_events",
    "generate_initial_data",
]
