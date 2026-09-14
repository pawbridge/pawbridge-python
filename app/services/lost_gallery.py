"""Repeatable, staged SAM 3 gallery builds; publish only a complete snapshot."""
from contextlib import nullcontext
import hashlib
import json
import re
from pathlib import Path

from PIL import Image, ImageOps
from app.services.dinov3 import DIMENSIONS, validate_vector
from app.services.sam3_focus import FOCUS_VERSION
from app.services.coat_color import VERSION as COLOR_VERSION, valid as valid_color
from app.services.gallery_prefetch import PhotoPrefetch

CONTRACT = "pawbridge-lost-gallery-v1"
PREFIX = "animals-lost-dinov3-sam3-"
METADATA = ("happen_date", "happen_place", "color", "special_mark", "description")


def read_manifest(path):
    with Path(path).open("rb") as source:
        payload = source.read(32 * 1024 * 1024 + 1)
    if len(payload) > 32 * 1024 * 1024:
        raise ValueError("Gallery manifest is too large")
    manifest = json.loads(payload)
    records = manifest.get("records", [])
    if (manifest.get("complete") is not True or not 1 <= len(records) <= 100_000):
        raise ValueError("A complete, nonempty snapshot is required")
    seen = set()
    for row in records:
        if (type(row.get("id")) is not int or row["id"] <= 0 or row["id"] in seen
                or row.get("species") not in {"DOG", "CAT"}
                or not re.fullmatch(r"[a-f0-9]{64}", row.get("source_sha256", ""))):
            raise ValueError("Invalid or duplicate gallery record")
        seen.add(row["id"])
        for field in METADATA:
            value = row.get(field)
            if value is not None and (not isinstance(value, str) or len(value) > 10000):
                raise ValueError("Invalid gallery metadata")
    canonical = json.dumps(sorted(records, key=lambda row: row["id"]), sort_keys=True,
                           separators=(",", ":"), ensure_ascii=False).encode()
    return records, hashlib.sha256(canonical).hexdigest()


def reusable_document(document, row):
    if (document.get("model_version") != FOCUS_VERSION
            or document.get("species") != row["species"]
            or document.get("source_sha256") != row["source_sha256"]):
        return False
    try:
        validate_vector(document.get("image_vector", []))
        if document.get("animal_vector") is not None:
            validate_vector(document["animal_vector"])
        return isinstance(document.get("focus_status"), str)
    except (RuntimeError, TypeError):
        return False


def reusable_color(document):
    return (document.get("coat_color_version") == COLOR_VERSION
            and (document.get("coat_color") is None or valid_color(document["coat_color"])))


def gallery_mapping(snapshot_hash):
    properties = {"id": {"type": "long"}, "species": {"type": "keyword"},
                  "source_sha256": {"type": "keyword"}, "model_version": {"type": "keyword"},
                  "focus_status": {"type": "keyword"},
                  "coat_color_version": {"type": "keyword"},
                  "coat_color": {"type": "object", "enabled": False},
                  "image_vector": {"type": "dense_vector", "dims": DIMENSIONS, "index": False},
                  "animal_vector": {"type": "dense_vector", "dims": DIMENSIONS, "index": False}}
    properties.update({key: {"type": "keyword", "index": False} for key in METADATA})
    return {"dynamic": "strict", "_meta": {"contract": CONTRACT, "model_version": FOCUS_VERSION,
                                             "snapshot_sha256": snapshot_hash, "coat_color_version": COLOR_VERSION}, "properties": properties}


def gallery_target(snapshot_hash):
    return PREFIX + "build-" + hashlib.sha256((FOCUS_VERSION + COLOR_VERSION + snapshot_hash).encode()).hexdigest()[:24]


class GalleryBuildCancelled(RuntimeError):
    pass


def build_gallery(es, encoder_factory, manifest_path, photo_root, alias, state_dir, progress=None, cancelled=None, photo_provider=None):
    # One GPU host is supported. All publishers share this state directory.
    import fcntl
    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    with (state / "gallery-publisher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _build_gallery(es, encoder_factory(), manifest_path, photo_root, alias, progress, cancelled, photo_provider)


def _build_gallery(es, encoder, manifest_path, photo_root, alias, progress=None, cancelled=None, photo_provider=None):
    def check_cancelled():
        if cancelled and cancelled():
            raise GalleryBuildCancelled("Gallery build stopped before publication")
    check_cancelled()
    if (not re.fullmatch(PREFIX + r"[a-z0-9][a-z0-9-]{0,60}", alias)
            or "-build-" in alias or encoder.model_version != FOCUS_VERSION):
        raise ValueError("SAM 3 gallery alias/model is required")
    records, snapshot_hash = read_manifest(manifest_path)
    root = Path(photo_root).resolve(strict=True)
    target = gallery_target(snapshot_hash)
    client = es.options(request_timeout=30, max_retries=0)
    if client.indices.exists_alias(name=alias):
        previous = client.indices.get_alias(name=alias)
        if len(previous) != 1:
            raise RuntimeError("Gallery alias must resolve to exactly one index")
        old_index = next(iter(previous))
        old_mapping = client.indices.get_mapping(index=old_index)[old_index]["mappings"]
        if (not old_index.startswith(PREFIX + "build-")
                or old_mapping.get("_meta", {}).get("contract") != CONTRACT):
            raise RuntimeError("Refusing to replace a gallery not owned by this builder")
    else:
        old_index = None
        if client.indices.exists(index=alias):
            raise RuntimeError("Gallery alias name is already a concrete index")
    expected_meta = gallery_mapping(snapshot_hash)["_meta"]
    if client.indices.exists(index=target):
        actual = client.indices.get_mapping(index=target)[target]["mappings"]
        if actual.get("_meta") != expected_meta:
            raise RuntimeError("Existing staging index has a different contract")
    else:
        client.indices.create(index=target, settings={"number_of_shards": 1, "number_of_replicas": 0},
                              mappings=gallery_mapping(snapshot_hash))
    check_cancelled()
    if old_index == target and client.count(index=target)["count"] == len(records):
        return {"alias": alias, "index": target, "records": len(records), "encoded": 0,
                "reused": len(records), "snapshot_sha256": snapshot_hash}
    encoded = reused = color_processed = color_available = 0
    for offset in range(0, len(records), 100):
        batch = records[offset:offset+100]
        ids = [str(row["id"]) for row in batch]
        cache = {}
        for source in dict.fromkeys(x for x in (old_index, target) if x):
            response = client.mget(index=source, ids=ids)
            for item in response["docs"]:
                if "error" in item:
                    raise RuntimeError("Gallery cache read failed")
                if item.get("found"):
                    cache[item["_source"]["id"]] = item["_source"]
        planned = []
        for row in batch:
            cached = cache.get(row["id"], {})
            reuse_vector = reusable_document(cached, row)
            needs_color = (reuse_vector and cached.get("focus_status") == "animal_mask"
                           and not reusable_color(cached))
            planned.append((row, cached, reuse_vector, needs_color))
        to_fetch = [row for row, _, reuse, color in planned if not reuse or color]
        check_cancelled()
        downloads = (PhotoPrefetch(to_fetch, photo_provider)
                     if photo_provider is not None and to_fetch else nullcontext())
        with downloads as prefetch:
            operations = []
            for row, cached, reuse_vector, needs_color in planned:
                check_cancelled()
                color = None
                if reuse_vector:
                    vector, animal, status = cached["image_vector"], cached.get("animal_vector"), cached["focus_status"]
                    reused += 1
                    if reusable_color(cached):
                        color = cached.get("coat_color")
                # A previously rejected mask has no usable foreground to backfill.
                if not reuse_vector or needs_color:
                    context = (prefetch.photo(row) if photo_provider is not None else
                               nullcontext(root / (row["source_sha256"] + ".image")))
                    with context as supplied_path:
                        path = Path(supplied_path).resolve(strict=True)
                        if not path.is_relative_to(root):
                            raise ValueError("Gallery photo escapes the configured root")
                        if path.stat().st_size > 10 * 1024 * 1024:
                            raise ValueError("Gallery photo exceeds 10MiB")
                        digest = hashlib.sha256()
                        with path.open("rb") as source:
                            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                                digest.update(chunk)
                        if digest.hexdigest() != row["source_sha256"]:
                            raise ValueError("Gallery photo hash mismatch")
                        with Image.open(path) as source:
                            # Some APMS JPEGs are MPO containers (primary + auxiliary photo).
                            # Match the evaluated gallery's primary-frame policy; no animation.
                            source.seek(0)
                            if (source.width * source.height > 16_000_000
                                    or (getattr(source, "n_frames", 1) != 1 and source.format != "MPO")):
                                raise ValueError("Gallery photo must have a bounded primary still image")
                            with ImageOps.exif_transpose(source).convert("RGB") as image:
                                if reuse_vector:
                                    color = encoder.describe_coat_color(image, row["species"])
                                else:
                                    embedding = encoder.encode_with_metadata(image, row["species"])
                                    if embedding.model_version != FOCUS_VERSION:
                                        raise RuntimeError("Encoder changed model during the build")
                                    vector, animal, status = embedding.vector, embedding.animal_vector, embedding.focus_status
                                    color = embedding.coat_color
                                    validate_vector(vector)
                                    if animal is not None:
                                        validate_vector(animal)
                                    encoded += 1
                    color_processed += 1
                if color is not None:
                    if not valid_color(color):
                        raise RuntimeError("Invalid foreground color descriptor")
                    color_available += 1
                document = {"id": row["id"], "species": row["species"], "source_sha256": row["source_sha256"],
                            "model_version": FOCUS_VERSION, "focus_status": status, "image_vector": vector,
                            "coat_color_version": COLOR_VERSION, "coat_color": color,
                            **{field: row.get(field) for field in METADATA}}
                if animal is not None:
                    document["animal_vector"] = animal
                operations.extend([{"index": {"_index": target, "_id": str(row["id"])}}, document])
        check_cancelled()
        response = client.bulk(operations=operations)
        if response.get("errors"):
            raise RuntimeError("Gallery bulk write failed; previous alias was preserved")
        if progress:
            progress({"processed": offset + len(batch), "total": len(records), "encoded": encoded, "reused": reused,
                      "color_processed": color_processed, "color_available": color_available})
    client.indices.refresh(index=target)
    count = client.count(index=target)["count"]
    if count != len(records):
        raise RuntimeError("Staging gallery count differs from the complete snapshot")
    # Detect a concurrent publisher before constructing the atomic alias update.
    current = list(client.indices.get_alias(name=alias)) if client.indices.exists_alias(name=alias) else []
    if current != ([old_index] if old_index else []):
        raise RuntimeError("Gallery alias changed during the build; retry explicitly")
    check_cancelled()
    if old_index != target:
        actions = ([{"remove": {"index": old_index, "alias": alias, "must_exist": True}}] if old_index else [])
        actions.append({"add": {"index": target, "alias": alias}})
        response = client.indices.update_aliases(actions=actions)
        if not response.get("acknowledged") or response.get("errors"):
            raise RuntimeError("Gallery publish acknowledgment is uncertain")
    if set(client.indices.get_alias(name=alias)) != {target}:
        raise RuntimeError("Published gallery alias does not match the completed index")
    return {"alias": alias, "index": target, "records": count, "encoded": encoded,
            "reused": reused, "color_processed": color_processed, "color_available": color_available,
            "snapshot_sha256": snapshot_hash}
