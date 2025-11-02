"""模拟数据源与实时 API 接口。

该模块提供：
- 基于 Faker 的离线数据批量生成方法；
- 模拟历史事件流的构造方法；
- 一个异步的 `MockRealtimeAPI`，用于按需拉取用户信息与实时事件。
- 基于 SQLite 的轻量持久化层，便于查询与扩展。

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
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine

# ============================================================================
# 默认配置常量
# ============================================================================
DEFAULT_LOCALE = "zh_CN"  # Faker 使用的语言环境（生成中文数据）
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "generated_data"  # 数据存储目录
DEFAULT_FILE_FORMAT: Literal["sqlite"] = "sqlite"  # 数据持久化格式（SQLite）
DEFAULT_DB_FILENAME = "mock_data.db"  # SQLite 数据库文件名
DEFAULT_USER_COUNT = 1_000  # 默认生成的用户数量
DEFAULT_PRODUCT_COUNT = 25  # 默认生成的产品数量
DEFAULT_APP_COUNT = 30  # 默认生成的 APP 数量
DEFAULT_AVG_EVENTS_PER_USER = 20  # 每用户平均生成的事件数

# ============================================================================
# 业务数据模板
# ============================================================================
# 套餐类型列表（运营商业务场景）
PLAN_TYPES = ["基础套餐", "5G畅享套餐", "家庭融合套餐", "校园流量套餐"]

# 用户等级分层
USER_LEVELS = ["普通用户", "银卡", "金卡", "白金"]

# 常见手机品牌
DEVICE_BRANDS = ["华为", "小米", "苹果", "OPPO", "vivo", "荣耀"]

# 常用 APP 名称（用于生成 APP 使用事件）
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

# 产品模板：(产品名称, 价格)
PRODUCT_TEMPLATES = [
    ("5G畅享套餐", 129.0),
    ("5G至尊套餐", 199.0),
    ("家庭宽带融合", 159.0),
    ("校园流量包", 69.0),
    ("国际漫游包", 299.0),
]

# ============================================================================
# 全局缓存
# ============================================================================
# SQLAlchemy 引擎缓存，避免重复创建连接
_ENGINE_CACHE: dict[Path, Engine] = {}


# ============================================================================
# 数据模型定义
# ============================================================================

class EventType(str, Enum):
    """用户行为事件类型枚举。
    
    定义系统支持的事件类型，用于构建异构图的不同边类型。
    
    Attributes:
        CALL: 通话事件（用户-用户边）
        ORDER: 订购事件（用户-产品边）
        APP_USAGE: APP使用事件（用户-APP边）
        CLICK: 点击事件（通用交互）
    """

    CALL = "通话"
    ORDER = "订购"
    APP_USAGE = "APP使用"
    CLICK = "点击"


@dataclass(slots=True)
class User:
    """用户基础画像信息。
    
    封装用户的静态属性，作为图神经网络的节点特征。
    
    Attributes:
        user_id (str): 用户唯一标识
        plan_type (str): 用户当前套餐类型
        monthly_fee (float): 月租费用（元）
        user_level (str): 用户等级（普通/银卡/金卡/白金）
        tenure_months (int): 在网时长（月）
        device_brand (str): 使用的设备品牌
    """

    user_id: str
    plan_type: str
    monthly_fee: float
    user_level: str
    tenure_months: int
    device_brand: str


@dataclass(slots=True)
class Product:
    """产品信息定义。
    
    表示可订购的产品/服务，作为图中的产品节点。
    
    Attributes:
        product_id (str): 产品唯一标识
        product_name (str): 产品名称
        price (float): 产品价格（元）
    """

    product_id: str
    product_name: str
    price: float


@dataclass(slots=True)
class App:
    """移动应用信息。
    
    表示移动应用，作为图中的 APP 节点。
    用户使用 APP 会产生 user-app 边。
    
    Attributes:
        app_id (str): APP 唯一标识
        app_name (str): APP 名称
    """

    app_id: str
    app_name: str


@dataclass(slots=True)
class Event:
    """用户行为事件。
    
    记录用户的行为轨迹，用于构建图边和生成推荐。
    不同事件类型对应不同的图边关系。
    
    Attributes:
        event_id (str): 事件唯一标识
        timestamp (datetime): 事件发生时间
        user_id (str): 触发事件的用户 ID
        event_type (EventType): 事件类型（通话/订购/APP使用/点击）
        target_user_id (str | None): 通话对象的用户 ID（仅通话事件）
        product_id (str | None): 关联的产品 ID（仅订购事件）
        app_id (str | None): 关联的 APP ID（仅 APP 使用事件）
        duration_seconds (int | None): 事件持续时长（秒），如通话时长
    """

    event_id: str
    timestamp: datetime
    user_id: str
    event_type: EventType
    target_user_id: str | None = None
    product_id: str | None = None
    app_id: str | None = None
    duration_seconds: int | None = None


# ============================================================================
# 数据生成函数
# ============================================================================

def generate_initial_data(
    num_users: int,
    num_products: int,
    num_apps: int,
    *,
    output_dir: str | Path | None = None,
    file_format: Literal["sqlite"] = DEFAULT_FILE_FORMAT,
) -> tuple[list[User], list[Product], list[App]]:
    """批量生成用户、产品与应用的静态模拟数据并持久化。
    
    该函数使用 Faker 库生成模拟数据，包括：
    - 用户：随机套餐、费用、等级、在网时长、设备品牌
    - 产品：基于模板或随机生成的产品名称和价格
    - APP：基于预定义列表或随机生成的 APP 名称
    
    生成的数据会保存到 SQLite 数据库中。
    
    Args:
        num_users (int): 要生成的用户数量，必须为正整数
        num_products (int): 要生成的产品数量，必须为正整数
        num_apps (int): 要生成的 APP 数量，必须为正整数
        output_dir (str | Path | None): 输出目录路径，None 时使用默认目录
        file_format (Literal["sqlite"]): 文件格式，当前仅支持 "sqlite"
        
    Returns:
        tuple[list[User], list[Product], list[App]]: 
            生成的用户列表、产品列表和 APP 列表
            
    Raises:
        ValueError: 当任何数量参数小于等于 0 时
        
    Examples:
        >>> users, products, apps = generate_initial_data(1000, 25, 30)
        >>> len(users)
        1000
    """
    # 参数验证
    if num_users <= 0 or num_products <= 0 or num_apps <= 0:
        raise ValueError("num_users, num_products, num_apps 必须为正整数")

    # 确保文件格式正确
    _ensure_sqlite_format(file_format)

    # 初始化 Faker 生成器（使用中文语言环境）
    faker = Faker(DEFAULT_LOCALE)
    output_path = Path(output_dir) if output_dir else DEFAULT_DATA_DIR
    _ensure_directory(output_path)

    # 生成用户数据
    users = [
        User(
            user_id=faker.uuid4(),                      # 唯一 ID
            plan_type=random.choice(PLAN_TYPES),        # 随机套餐类型
            monthly_fee=round(random.uniform(39, 399), 2),  # 月费：39-399元
            user_level=random.choice(USER_LEVELS),      # 随机用户等级
            tenure_months=random.randint(1, 180),       # 在网时长：1-180月
            device_brand=random.choice(DEVICE_BRANDS),  # 随机设备品牌
        )
        for _ in range(num_users)
    ]

    # 生成产品数据
    # 优先使用预定义模板，超出部分随机生成
    products = [
        Product(
            product_id=faker.uuid4(),
            # 前 N 个使用模板名称，其余随机生成
            product_name=name if idx < len(PRODUCT_TEMPLATES) else faker.catch_phrase(),
            # 前 N 个使用模板价格，其余随机生成
            price=round(price if idx < len(PRODUCT_TEMPLATES) else random.uniform(49, 499), 2),
        )
        for idx, (name, price) in enumerate(random.choices(PRODUCT_TEMPLATES, k=num_products))
    ]

    # 生成 APP 数据
    # 前 10 个使用预定义名称，其余随机生成公司名
    apps = [
        App(
            app_id=faker.uuid4(),
            app_name=APP_NAMES[idx] if idx < len(APP_NAMES) else faker.company(),
        )
        for idx in range(num_apps)
    ]

    # 持久化到 SQLite 数据库
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
    file_format: Literal["sqlite"] = DEFAULT_FILE_FORMAT,
    avg_events_per_user: int = DEFAULT_AVG_EVENTS_PER_USER,
) -> list[Event]:
    """为给定用户集合生成历史事件序列并持久化。
    
    该函数模拟用户在指定时间段内的行为轨迹，生成多种类型的事件：
    - 通话事件：用户之间的通话（user-user 边）
    - 订购事件：用户购买产品（user-product 边）
    - APP 使用事件：用户使用 APP（user-app 边）
    - 点击事件：用户点击产品或 APP
    
    每个用户生成的事件数量服从高斯分布（均值=avg_events_per_user）。
    
    Args:
        users (Iterable[User]): 用户列表或 DataFrame
        products (Iterable[Product]): 产品列表或 DataFrame
        apps (Iterable[App]): APP 列表或 DataFrame
        start_date (datetime): 事件生成的开始时间
        end_date (datetime): 事件生成的结束时间
        output_dir (str | Path | None): 输出目录路径
        file_format (Literal["sqlite"]): 文件格式
        avg_events_per_user (int): 每用户平均事件数，实际数量会有正态分布的波动
        
    Returns:
        list[Event]: 按时间排序的事件列表
        
    Raises:
        ValueError: 当 start_date >= end_date 或 avg_events_per_user <= 0 时
        
    Examples:
        >>> events = generate_historical_events(
        ...     users=users,
        ...     products=products,
        ...     apps=apps,
        ...     start_date=datetime(2024, 1, 1),
        ...     end_date=datetime(2024, 1, 31),
        ...     avg_events_per_user=20
        ... )
    """
    # 参数验证
    if start_date >= end_date:
        raise ValueError("start_date 必须早于 end_date")
    if avg_events_per_user <= 0:
        raise ValueError("avg_events_per_user 必须为正整数")

    _ensure_sqlite_format(file_format)

    # 初始化生成器和路径
    faker = Faker(DEFAULT_LOCALE)
    output_path = Path(output_dir) if output_dir else DEFAULT_DATA_DIR
    _ensure_directory(output_path)

    # 确保输入是列表格式（支持 DataFrame 输入）
    user_list = _ensure_sequence(users, _row_to_user)
    product_list = _ensure_sequence(products, _row_to_product)
    app_list = _ensure_sequence(apps, _row_to_app)

    # 为每个用户生成事件
    events: list[Event] = []
    for user in user_list:
        # 事件数量服从高斯分布，标准差为均值的 1/3
        event_count = max(1, int(random.gauss(avg_events_per_user, avg_events_per_user // 3 or 1)))
        
        for _ in range(event_count):
            # 随机选择事件类型
            event_type = random.choice(list(EventType))
            # 在时间范围内随机选择时间戳
            timestamp = faker.date_time_between_dates(datetime_start=start_date, datetime_end=end_date)
            
            # 根据事件类型初始化关联字段
            duration: int | None = None
            target_user_id: str | None = None
            product_id: str | None = None
            app_id: str | None = None

            # 根据事件类型填充相应字段
            if event_type is EventType.CALL:
                # 通话事件：随机选择另一个用户作为通话对象
                other_user = random.choice(user_list)
                target_user_id = other_user.user_id if other_user.user_id != user.user_id else None
                duration = random.randint(30, 1800)  # 通话时长：30秒-30分钟
                
            elif event_type is EventType.ORDER:
                # 订购事件：随机选择一个产品
                product_id = random.choice(product_list).product_id
                
            elif event_type is EventType.APP_USAGE:
                # APP 使用事件：随机选择一个 APP
                app_id = random.choice(app_list).app_id
                duration = random.randint(60, 3600)  # 使用时长：1分钟-1小时
                
            elif event_type is EventType.CLICK:
                # 点击事件：50% 点击产品，50% 点击 APP
                if random.random() < 0.5:
                    product_id = random.choice(product_list).product_id
                else:
                    app_id = random.choice(app_list).app_id

            # 创建事件对象
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

    # 按时间排序，模拟真实的事件流
    events.sort(key=lambda event: event.timestamp)
    
    # 持久化到 SQLite
    _save_dataframe(
        pd.DataFrame([_event_to_record(event) for event in events]),
        output_path,
        "events",
        file_format,
    )
    
    return events


# ============================================================================
# 模拟实时 API 类
# ============================================================================

class MockRealtimeAPI:
    """模拟实时数据接口，兼容未来真实数据服务的契约。
    
    该类提供：
    - 用户、产品、APP 的查询接口（异步）
    - 实时事件流的生成与推送（异步迭代器）
    - 基于 SQLite 的数据持久化
    - 自动初始化种子数据（首次运行时）
    
    设计目标：
    - 与后端服务解耦，便于未来替换为真实数据源
    - 支持异步操作，兼容 asyncio 生态
    - 模拟真实场景的事件流速率控制
    
    Attributes:
        data_dir (Path): 数据存储目录
        file_format (str): 数据文件格式（当前仅支持 "sqlite"）
        realtime_delay (float): 事件推送间隔（秒），用于速率控制
    """

    def __init__(
        self,
        *,
        data_dir: str | Path | None = None,
        file_format: Literal["sqlite"] = DEFAULT_FILE_FORMAT,
        realtime_delay_seconds: float = 1.0,
        user_count: int = DEFAULT_USER_COUNT,
        product_count: int = DEFAULT_PRODUCT_COUNT,
        app_count: int = DEFAULT_APP_COUNT,
        avg_events_per_user: int = DEFAULT_AVG_EVENTS_PER_USER,
        history_days: int = 30,
    ) -> None:
        """构造模拟实时 API。

        Args:
            data_dir (str | Path | None): 
                生成与存放模拟数据集的目录，None 时使用默认目录
            file_format (Literal["sqlite"]): 
                底层数据格式，当前仅支持 "sqlite"
            realtime_delay_seconds (float): 
                事件推送延迟（秒），通过 asyncio.sleep 控制推送速率
                用于模拟准实时场景，避免过快消费
            user_count (int): 
                初始化时生成的用户数量
            product_count (int): 
                初始化时生成的产品数量
            app_count (int): 
                初始化时生成的 APP 数量
            avg_events_per_user (int): 
                每用户平均生成的历史事件数
            history_days (int): 
                历史事件的时间跨度（天），从当前时间向前推算
                
        Note:
            首次创建实例时会自动检查数据是否存在，
            如果不存在会调用 _ensure_seed_data() 生成种子数据。
        """
        # 初始化 Faker 生成器（用于生成随机数据）
        self._faker = Faker(DEFAULT_LOCALE)
        
        # 数据目录和格式设置
        self._data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        _ensure_sqlite_format(file_format)
        self._file_format = file_format
        
        # 实时推送速率控制（避免事件推送过快）
        self._realtime_delay = max(0.0, realtime_delay_seconds)
        
        # 数据生成配置参数（存储为实例变量，支持动态调整）
        self._user_count = user_count
        self._product_count = product_count
        self._app_count = app_count
        self._avg_events_per_user = avg_events_per_user
        self._history_days = history_days
        self._engine = _get_engine(self._data_dir)

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
    def file_format(self) -> Literal["sqlite"]:
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

        users_exists = _table_exists(self._engine, "users")
        products_exists = _table_exists(self._engine, "products")
        apps_exists = _table_exists(self._engine, "apps")
        events_exists = _table_exists(self._engine, "events")

        # 检查表是否存在且有数据
        need_initial = not (users_exists and products_exists and apps_exists)
        if not need_initial and users_exists:
            # 表存在，但需要检查是否为空
            import pandas as pd
            try:
                users_count = pd.read_sql_query("SELECT COUNT(*) as cnt FROM users", self._engine).iloc[0]['cnt']
                if users_count == 0:
                    need_initial = True
            except Exception:
                need_initial = True
        
        if need_initial:
            users, products, apps = generate_initial_data(
                self._user_count,
                self._product_count,
                self._app_count,
                output_dir=self._data_dir,
                file_format=self._file_format,
            )
        else:
            users = products = apps = None

        # 检查events表是否存在且有数据
        need_events = not events_exists
        if not need_events and events_exists:
            # 表存在，但需要检查是否为空
            import pandas as pd
            try:
                events_count = pd.read_sql_query("SELECT COUNT(*) as cnt FROM events", self._engine).iloc[0]['cnt']
                if events_count == 0:
                    need_events = True
            except Exception:
                need_events = True
        
        if need_events:
            if users is None or products is None or apps is None:
                users_df = _load_dataframe(self._data_dir, "users", self._file_format)
                products_df = _load_dataframe(self._data_dir, "products", self._file_format)
                apps_df = _load_dataframe(self._data_dir, "apps", self._file_format)
                users = [_row_to_user(row) for _, row in users_df.iterrows()]
                products = [_row_to_product(row) for _, row in products_df.iterrows()]
                apps = [_row_to_app(row) for _, row in apps_df.iterrows()]

            now = datetime.utcnow()
            generate_historical_events(
                users,
                products,
                apps,
                start_date=now - timedelta(days=self._history_days),
                end_date=now,
                output_dir=self._data_dir,
                file_format=self._file_format,
                avg_events_per_user=self._avg_events_per_user,
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
        _append_to_table(record_df, self._data_dir, "events")
        self._events_df = pd.concat([self._events_df, record_df], ignore_index=True)


def _database_path(data_dir: Path) -> Path:
    """返回 SQLite 数据文件路径。"""
    return data_dir / DEFAULT_DB_FILENAME


def _get_engine(data_dir: Path) -> Engine:
    """复用指定目录对应的 SQLite Engine。"""
    db_path = _database_path(data_dir)
    engine = _ENGINE_CACHE.get(db_path)
    if engine is None:
        _ensure_directory(data_dir)
        engine = create_engine(f"sqlite:///{db_path.as_posix()}")
        _ENGINE_CACHE[db_path] = engine
    return engine


def _table_exists(engine: Engine, table_name: str) -> bool:
    """检测目标表是否已经存在于数据库中。"""
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def _append_to_table(df: pd.DataFrame, data_dir: Path, name: str) -> None:
    """向目标表追加记录，保持与实时事件存储一致。"""
    engine = _get_engine(data_dir)
    df.to_sql(name, engine, if_exists="append", index=False)


def _ensure_sqlite_format(file_format: str) -> None:
    """统一校验入参，避免误用旧的文件落盘逻辑。"""
    if file_format != DEFAULT_FILE_FORMAT:
        raise ValueError("仅支持 SQLite 存储后端，请将 file_format 设置为 'sqlite'")


def _ensure_directory(path: Path) -> None:
    """创建目录并忽略已存在场景，供数据生成与持久化使用。"""
    path.mkdir(parents=True, exist_ok=True)


def _save_dataframe(
    df: pd.DataFrame,
    data_dir: Path,
    name: str,
    file_format: Literal["sqlite"],
) -> Path:
    """将 ``DataFrame`` 内容刷新写入 SQLite 表。"""
    _ensure_sqlite_format(file_format)
    engine = _get_engine(data_dir)
    serializable_df = df.copy()
    for column in serializable_df.columns:
        if column == "timestamp" or pd.api.types.is_datetime64_any_dtype(serializable_df[column]):
            serializable_df[column] = pd.to_datetime(serializable_df[column])

    serializable_df.to_sql(name, engine, if_exists="replace", index=False)
    return _database_path(data_dir)


def _load_dataframe(
    data_dir: Path,
    name: str,
    file_format: Literal["sqlite"],
) -> pd.DataFrame:
    """读取目标 SQLite 表并返回 ``DataFrame``。"""
    _ensure_sqlite_format(file_format)
    engine = _get_engine(data_dir)
    if not _table_exists(engine, name):
        raise FileNotFoundError(f"未找到数据表: {name}")

    parse_dates = ["timestamp"] if name == "events" else None
    return pd.read_sql_table(name, engine, parse_dates=parse_dates)


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


def load_dataset_dataframe(
    name: str,
    data_dir: str | Path | None = None,
    *,
    file_format: Literal["sqlite"] = DEFAULT_FILE_FORMAT,
) -> pd.DataFrame:
    """提供给其他模块的统一读表入口，默认从生成目录加载。"""
    target_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    return _load_dataframe(target_dir, name, file_format)


__all__ = [
    "App",
    "Event",
    "EventType",
    "MockRealtimeAPI",
    "Product",
    "User",
    "generate_historical_events",
    "generate_initial_data",
    "load_dataset_dataframe",
]
