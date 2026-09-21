"""Explicit backend selection; import does not open a pool or contact a database."""
from contextlib import contextmanager
import os
import threading

_lock = threading.Lock()
_store = None


def backend():
    value = os.getenv('LOST_STORAGE_BACKEND', 'elasticsearch')
    if value not in {'elasticsearch', 'postgresql'}:
        raise RuntimeError('Unknown lost-search storage backend')
    return value


def get_postgresql_store():
    global _store
    if backend() != 'postgresql':
        raise RuntimeError('PostgreSQL backend was not selected')
    with _lock:
        if _store is None:
            from psycopg_pool import ConnectionPool
            from app.services.pg_gallery_store import PostgresqlGalleryStore
            dsn = os.getenv('LOST_PG_DSN')
            size = int(os.getenv('LOST_PG_POOL_MAX_SIZE', '2'))
            if not dsn or not 1 <= size <= 4:
                raise RuntimeError('PostgreSQL DSN and a pool limit between 1 and 4 are required')
            pool = ConnectionPool(dsn, min_size=0, max_size=size, max_waiting=4, timeout=3,
                                  kwargs={'autocommit': True, 'connect_timeout': 3,
                                          'application_name': 'pawbridge-lost-vector'},
                                  max_idle=60, max_lifetime=600, reconnect_timeout=5,
                                  check=ConnectionPool.check_connection, open=False,
                                  name='lost-vector', num_workers=1)
            pool.open()
            _store = PostgresqlGalleryStore(pool)
        return _store


def close_postgresql_store():
    global _store
    with _lock:
        if _store is not None:
            _store.pool.close()
            _store = None


@contextmanager
def storage_session():
    try:
        yield get_postgresql_store() if backend() == 'postgresql' else None
    finally:
        close_postgresql_store()
