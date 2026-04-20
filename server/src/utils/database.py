from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from src.models.database import Base
import src.models.customer
import src.models.product  
import src.models.order  
import src.models.ticket  
import src.models.agent_log  
from src.utils.config import get_settings

settings = get_settings()

database_url = settings.DATABASE_URL

try:
    engine = create_async_engine(
        database_url,
        echo=False,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
        connect_args={
            "statement_cache_size": 0,
            "prepared_statement_cache_size": 0,
        },
    )
except Exception as e:
    raise

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db():
    try:
        async with async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                raise
    except Exception as e:
        raise