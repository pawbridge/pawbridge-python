"""ES gallery persistence; inference/downloads never run inside a storage operation."""
from app.services.lost_gallery import CONTRACT, PREFIX, gallery_mapping


class ElasticsearchGalleryStore:
    def __init__(self, es):
        self.client = es.options(request_timeout=30, max_retries=0)

    def begin(self, alias, target, snapshot_hash, total):
        client = self.client
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
        return old_index

    def documents(self, old_index, target, ids):
        client = self.client
        cache = {}
        for source in dict.fromkeys(x for x in (old_index, target) if x):
            response = client.mget(index=source, ids=ids)
            for item in response["docs"]:
                if "error" in item:
                    raise RuntimeError("Gallery cache read failed")
                if item.get("found"):
                    cache[item["_source"]["id"]] = item["_source"]
        return cache

    def completed(self, target, total):
        # ES does not persist a separate completion flag for an unaliased generation.
        return False

    def count(self, target):
        return self.client.count(index=target)["count"]

    def write_page(self, target, documents):
        operations = []
        for document in documents:
            operations.extend([{"index": {"_index": target, "_id": str(document["id"])}}, document])
        if self.client.bulk(operations=operations).get("errors"):
            raise RuntimeError("Gallery bulk write failed; previous alias was preserved")

    def publish(self, alias, target, old_index, total, check_cancelled):
        client = self.client
        client.indices.refresh(index=target)
        count = client.count(index=target)["count"]
        if count != total:
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
        return count
