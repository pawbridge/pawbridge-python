"""Read-only pre-GPU storage check. Does not load models, mutate data or print secrets."""
import json
import os
import sys
import urllib.request


def check_elasticsearch():
    endpoint = os.environ["ES_URL"].rstrip("/")
    with urllib.request.urlopen(endpoint + "/_cluster/health?wait_for_status=yellow&timeout=5s", timeout=8) as response:
        health = json.load(response)
    if health.get("status") not in {"green", "yellow"} or health.get("timed_out", False):
        raise RuntimeError("Elasticsearch is not ready")


def check_postgresql():
    import psycopg
    dsn = os.environ["LOST_PG_DSN"]
    if not dsn.strip():
        raise ValueError("PostgreSQL DSN required")
    # One short connection, closed before model loading. Options override DSN timeouts.
    with psycopg.connect(dsn, connect_timeout=3, application_name="pawbridge-storage-preflight",
                         options="-c default_transaction_read_only=on -c statement_timeout=3000 -c lock_timeout=1000") as connection:
        if connection.execute("SELECT current_database()").fetchone() != ("pawbridge",):
            raise RuntimeError("Unexpected PostgreSQL database")
        role = connection.execute("SELECT rolsuper,rolcreatedb,rolcreaterole,rolreplication FROM pg_roles WHERE rolname=current_user").fetchone()
        if role != (False, False, False, False):
            raise RuntimeError("Restricted vector role required")
        dimensions = connection.execute("SELECT a.attname,a.atttypmod FROM pg_attribute a "
            "JOIN pg_type t ON t.oid=a.atttypid JOIN pg_namespace n ON n.oid=t.typnamespace "
            "WHERE a.attrelid='pawbridge_animal.lost_gallery_documents'::regclass "
            "AND a.attname IN ('image_vector','animal_vector') AND NOT a.attisdropped "
            "AND t.typname='vector' AND n.nspname='public' ORDER BY a.attname").fetchall()
        if dimensions != [("animal_vector", 1024), ("image_vector", 1024)]:
            raise RuntimeError("1024-dimensional gallery schema required")
        # Resolve columns and read privileges without scanning gallery vectors.
        connection.execute("SELECT h.alias,h.build_key,b.metadata,b.completed,b.expected_count,d.document,"
            "d.image_vector,d.animal_vector FROM pawbridge_animal.lost_gallery_heads h "
            "JOIN pawbridge_animal.lost_gallery_builds b ON b.build_key=h.build_key "
            "JOIN pawbridge_animal.lost_gallery_documents d ON d.build_key=b.build_key LIMIT 0")
        connection.execute("SELECT id,species,status FROM pawbridge_animal.animals LIMIT 0")
    # Published-generation/model validation remains the application's responsibility.


def main():
    backend = os.getenv("LOST_STORAGE_BACKEND", "elasticsearch")
    try:
        if backend == "elasticsearch":
            check_elasticsearch()
        elif backend == "postgresql":
            check_postgresql()
        else:
            raise ValueError("Unknown storage backend")
    except Exception:
        # Driver/HTTP exception text may contain DSNs, credentials or signed URLs.
        print("Gallery storage preflight failed; verify selected backend, access and schema", file=sys.stderr)
        return 1
    print("Gallery storage preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
