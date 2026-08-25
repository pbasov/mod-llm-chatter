"""Hindsight-backed long-term memory for companions.

llm_bot_memories is a flat table: three memories picked with
ORDER BY RAND(), capped at thirty, evicted at random. There is
no way to recall something *relevant*, which is the one thing
a companion's memory needs to do.

Hindsight (already running on the fleet, already reachable
from this cluster) gives semantic recall over pgvector, an
entity graph, and observation consolidation. This module
mirrors memories into it and reads them back.

Two things shape the design:

  * Recall costs a flat ~5s regardless of budget or token
    count -- it is embed + vector search + cross-encoder
    rerank, and it does not tune down. It can therefore never
    sit inline in a reply. Retain is fired async and recall is
    prefetched in the background at group join.

  * One realm bank, not one per bot. Per-bot banks would
    fragment the entity graph into dozens of isolated copies
    of the same world, with the player a separate unlinked
    entity in each. Isolation is by tag instead, and recall
    is scoped so a bot can only ever reach its own experience
    with this player -- a bot narrating someone else's memory
    in the first person is exactly the failure mode we are
    trying to remove.

Uses stdlib http only; the bridge image carries no requests.
"""

import json
import logging
import threading
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_DEFAULT_BASE = (
    'http://hindsight-api-global.hindsight.svc.cluster.local:8888'
)

_bank_lock = threading.Lock()
_bank_ready = set()

_rate_lock = threading.Lock()
_rate_window = []

_prefetch_lock = threading.Lock()
_prefetch_cache = {}
_PREFETCH_TTL = 1800


def _enabled(config):
    try:
        return bool(int(config.get(
            'LLMChatter.Hindsight.Enable', 0)))
    except (ValueError, TypeError):
        return False


def _base(config):
    return config.get(
        'LLMChatter.Hindsight.BaseUrl', _DEFAULT_BASE
    ).rstrip('/')


def _bank(config):
    return config.get(
        'LLMChatter.Hindsight.Bank', 'azerothcore')


def _timeout(config):
    try:
        return float(config.get(
            'LLMChatter.Hindsight.RecallTimeout', 8))
    except (ValueError, TypeError):
        return 8.0


def _request(config, method, path, payload=None,
             timeout=None):
    """One JSON call. Returns parsed body or None."""
    url = _base(config) + path
    data = None
    if payload is not None:
        data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(
            req, timeout=timeout or _timeout(config)
        ) as resp:
            body = resp.read()
        return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = ''
        try:
            detail = exc.read().decode('utf-8')[:300]
        except Exception:
            pass
        logger.warning(
            "Hindsight %s %s -> %s %s",
            method, path, exc.code, detail,
        )
    except Exception as exc:
        logger.warning(
            "Hindsight %s %s failed: %s", method, path, exc)
    return None


_RETAIN_MISSION = (
    "These are first-person journal entries written by "
    "companion characters adventuring in World of Warcraft "
    "(Wrath of the Lich King, 3.3.5a). Extract who was "
    "involved, what happened, where it happened, and who they "
    "were with. Treat character names, zones, dungeons, "
    "bosses, quests and items as entities. Do not invent "
    "lore, and never introduce places or events from "
    "expansions later than Wrath of the Lich King."
)


def _ensure_bank(config):
    """Create the realm bank once per process."""
    bank = _bank(config)
    with _bank_lock:
        if bank in _bank_ready:
            return True
        ok = _request(
            config, 'PUT', '/v1/default/banks/%s' % bank,
            {
                'name': bank,
                'retain_mission': _RETAIN_MISSION,
                'enable_observations': True,
            },
            timeout=15,
        )
        if ok is not None:
            _bank_ready.add(bank)
            logger.info("Hindsight bank '%s' ready", bank)
            return True
        return False


def _rate_ok(config):
    """Token bucket over retains.

    Hindsight extracts facts with an LLM call, and on this
    fleet that is the *same* vLLM the bridge talks to. Without
    a ceiling a busy session would have companions competing
    with themselves for GPU slots.
    """
    try:
        limit = int(config.get(
            'LLMChatter.Hindsight.MaxRetainPerMinute', 20))
    except (ValueError, TypeError):
        limit = 20
    if limit <= 0:
        return False
    now = time.time()
    with _rate_lock:
        while _rate_window and now - _rate_window[0] > 60:
            _rate_window.pop(0)
        if len(_rate_window) >= limit:
            return False
        _rate_window.append(now)
        return True


def _tags(ctx, memory_type=None, zone=None):
    tags = []
    if ctx.get('bot_name'):
        tags.append('bot:%s' % ctx['bot_name'])
    if ctx.get('player_name'):
        tags.append('player:%s' % ctx['player_name'])
    if memory_type:
        tags.append('type:%s' % memory_type)
    if zone:
        tags.append('zone:%s' % zone)
    return tags


def retain(config, ctx, items):
    """Mirror memories into Hindsight. Fire and forget.

    items: [{'id', 'memory', 'memory_type', 'zone',
             'session_start'}]
    """
    if not _enabled(config) or not items:
        return False
    if not ctx.get('bot_name'):
        return False
    if not _ensure_bank(config):
        return False

    payload_items = []
    for it in items:
        text = (it.get('memory') or '').strip()
        if not text:
            continue
        if not _rate_ok(config):
            logger.info(
                "Hindsight retain rate limit reached; "
                "%d memories left in MySQL only",
                len(items) - len(payload_items),
            )
            break
        item = {
            'content': text,
            'tags': _tags(
                ctx, it.get('memory_type'), it.get('zone')),
            'metadata': {
                'bot_guid': ctx.get('bot_guid'),
                'player_guid': ctx.get('player_guid'),
                'memory_type': it.get('memory_type'),
                'zone': it.get('zone'),
            },
        }
        if it.get('id'):
            # Keyed to the MySQL row so a replay is an update,
            # not a duplicate.
            item['document_id'] = 'acmem-%s' % it['id']
        ts = it.get('session_start')
        if ts:
            try:
                item['timestamp'] = time.strftime(
                    '%Y-%m-%dT%H:%M:%S',
                    time.gmtime(float(ts)))
            except (ValueError, TypeError, OSError):
                pass
        payload_items.append(item)

    if not payload_items:
        return False

    result = _request(
        config, 'POST',
        '/v1/default/banks/%s/memories' % _bank(config),
        {'items': payload_items, 'async': True},
        timeout=15,
    )
    if result is None:
        return False
    logger.debug(
        "Hindsight retained %d memories for %s",
        len(payload_items), ctx.get('bot_name'))
    return True


def recall_memories(config, ctx, topic, limit=4):
    """Semantic recall, scoped to this bot and this player.

    tag_groups is used rather than the top-level tags filter
    on purpose: tags_match defaults to 'any', documented as
    "OR, includes untagged", so untagged memories would leak
    into every result. TagGroupLeaf defaults to any_strict and
    we ask for all_strict explicitly, which excludes them.
    """
    if not _enabled(config):
        return []
    if not ctx.get('bot_name') or not ctx.get('player_name'):
        return []
    if not (topic or '').strip():
        return []

    groups = [
        {'tags': ['bot:%s' % ctx['bot_name']],
         'match': 'all_strict'},
        {'tags': ['player:%s' % ctx['player_name']],
         'match': 'all_strict'},
    ]
    result = _request(
        config, 'POST',
        '/v1/default/banks/%s/memories/recall'
        % _bank(config),
        {
            'query': str(topic)[:300],
            'tag_groups': groups,
            'max_tokens': 400,
        },
    )
    if not result:
        return []

    out = []
    for row in (result.get('results') or [])[:limit]:
        text = ''
        if isinstance(row, dict):
            text = (row.get('content')
                    or row.get('text')
                    or row.get('fact')
                    or row.get('memory') or '')
            if not text:
                # Unknown shape: keep it readable rather than
                # dumping a dict into a prompt.
                text = json.dumps(row)[:300]
        else:
            text = str(row)
        text = ' '.join(str(text).split())
        if text:
            out.append(text[:300])
    return out


def prefetch(config, ctx, topic):
    """Warm the recall cache on a background thread.

    Called at group join, where a ~5s round trip disappears
    into the join sequence's existing delays.
    """
    if not _enabled(config):
        return
    key = (ctx.get('bot_guid'), ctx.get('player_guid'))
    if not all(key):
        return

    def _work():
        try:
            hits = recall_memories(config, ctx, topic)
            if hits:
                with _prefetch_lock:
                    _prefetch_cache[key] = (time.time(), hits)
        except Exception:
            logger.debug(
                "Hindsight prefetch failed", exc_info=True)

    threading.Thread(
        target=_work, name='hindsight-prefetch', daemon=True
    ).start()


def take_prefetched(ctx):
    """Consume a warmed recall, if one landed in time."""
    key = (ctx.get('bot_guid'), ctx.get('player_guid'))
    with _prefetch_lock:
        entry = _prefetch_cache.get(key)
        if not entry:
            return []
        stamped, hits = entry
        if time.time() - stamped > _PREFETCH_TTL:
            _prefetch_cache.pop(key, None)
            return []
        return list(hits)


def retain_activated(db, config, group_id, bot_guid,
                     player_guid, session_start):
    """Mirror the memories a farewell just committed.

    Runs on the memory executor thread that already did the
    activation, and the POST is async on Hindsight's side, so
    this does not sit in front of anything the player waits
    for.
    """
    if not _enabled(config):
        return
    try:
        cur = db.cursor(dictionary=True)
        cur.execute(
            "SELECT id, memory, memory_type, session_start"
            " FROM llm_bot_memories"
            " WHERE group_id = %s AND bot_guid = %s"
            "   AND active = 1 AND session_start = %s",
            (group_id, bot_guid, session_start),
        )
        rows = cur.fetchall()
        if not rows:
            return

        cur.execute(
            "SELECT bot_name FROM llm_bot_identities"
            " WHERE bot_guid = %s", (bot_guid,))
        ident = cur.fetchone() or {}
        bot_name = ident.get('bot_name')
        if not bot_name:
            cur.execute(
                "SELECT name FROM characters WHERE guid = %s",
                (bot_guid,))
            bot_name = (cur.fetchone() or {}).get('name')

        cur.execute(
            "SELECT name FROM characters WHERE guid = %s",
            (player_guid,))
        player_name = (cur.fetchone() or {}).get('name')

        if not bot_name or not player_name:
            return

        ctx = {
            'bot_guid': bot_guid,
            'bot_name': bot_name,
            'player_guid': player_guid,
            'player_name': player_name,
        }
        retain(config, ctx, rows)
    except Exception:
        logger.debug(
            "Hindsight retain_activated failed", exc_info=True)
