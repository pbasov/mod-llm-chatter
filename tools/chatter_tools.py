"""On-demand game lookups for bots (tool calling).

Everything a bot knows normally arrives by push: the
worldserver drops an event row with a fixed payload and the
bridge folds it into one prompt. That works for reacting to
what just happened and fails completely for anything the
player asks about -- quest logs, items, NPCs, chains.

This module gives the model a way to ask. It deliberately
does NOT build a tool layer from scratch: mod-llm-guide
already ships GameToolExecutor, 29 tools over the world DB
whose only dependency is a mysql-connector kwargs dict, and
both tools/ trees are present in the same bridge image. We
borrow it and add the two lookups it has no equivalent for.

The lookup happens in a separate grounding call (see
chatter_llm.call_llm_with_tools). The speaking call that
follows is byte-identical to an ungrounded one apart from a
<lookup_results> block, so the JSON envelope, the parser
cascade, pacing and delivery are all untouched.
"""

import logging
import re
import sys
import threading

logger = logging.getLogger(__name__)

# "*" = every tool mod-llm-guide exposes. A curated subset was the
# original instinct, to save prompt tokens on the grounding call --
# but a tool that is absent is not a saving, it is a hole the model
# fills from memory. With no boss-listing tool available it named
# Blackwing Lair's FIRST boss as the final boss of the Deadmines,
# and then defended the answer. The schemas ride only on the small
# Stage-A prompt, never on the speaking call, so breadth is far
# cheaper than being confidently wrong.
DEFAULT_ALLOWED = '*'

# Guide's tools emit link markers for its own C++ converter,
# which chatter does not have. Left in, bots type raw markup
# into party chat.
_LINK_MARKER_RE = re.compile(r'\[\[[a-z]+:\d+:([^\]]+)\]\]')

_MAX_RESULT_CHARS = 1200

# Guide's tool descriptions instruct the model to reproduce
# [[npc:ID:Name]] markers verbatim, because guide's C++ turns
# them into clickable links. Chatter has no such converter, so
# any sentence selling the markers has to go or bots type raw
# markup into party chat.
_ANY_MARKER_RE = re.compile(r'\[\[[^\]]*\]\]')
_MARKER_SENTENCE_RE = re.compile(
    r'[^.!?]*(?:marker|colored link|as-is)[^.!?]*[.!?]\s*',
    re.IGNORECASE,
)

_guide_lock = threading.Lock()
_guide_mod = None
_guide_tried = False


def strip_link_markers(text):
    """Turn [[npc:123:Hogger]] into plain Hogger."""
    if not text:
        return ''
    return _LINK_MARKER_RE.sub(r'\1', text)


def _load_guide(config):
    """Import mod-llm-guide's game_tools, or return None.

    Note the path is APPENDED, not prepended. The two trees
    both contain a spell_names module; they expose the same
    SPELL_NAMES/SPELL_DESCRIPTIONS int->str maps, but
    chatter's is a small JSON loader while guide's is an
    81k-line literal. Appending means chatter's wins the name
    and we never parse the big one.
    """
    global _guide_mod, _guide_tried
    with _guide_lock:
        if _guide_tried:
            return _guide_mod
        _guide_tried = True
        path = config.get(
            'LLMChatter.Tools.GuidePath', '/app/guide'
        )
        if path not in sys.path:
            sys.path.append(path)
        try:
            import game_tools
            _guide_mod = game_tools
            logger.info(
                "Loaded mod-llm-guide tool layer from %s "
                "(%d tools available)",
                path, len(game_tools.GAME_TOOLS),
            )
        except Exception as exc:
            logger.warning(
                "mod-llm-guide tool layer unavailable at %s "
                "(%s); bots keep only their native lookups",
                path, exc,
            )
            _guide_mod = None
        return _guide_mod


def build_executor(config, ctx=None):
    """GameToolExecutor wired to chatter's DB credentials.

    Returns None when the guide layer is missing, which
    cleanly reduces the toolset to the native lookups rather
    than failing the reply.
    """
    guide = _load_guide(config)
    if guide is None:
        return None
    db_config = {
        'host': config.get(
            'LLMChatter.Database.Host', 'localhost'),
        'port': int(config.get(
            'LLMChatter.Database.Port', 3306)),
        'user': config.get(
            'LLMChatter.Database.User', 'acore'),
        'password': config.get(
            'LLMChatter.Database.Password', ''),
        'database': 'acore_world',
    }
    try:
        executor = guide.GameToolExecutor(db_config)
    except Exception:
        logger.error(
            "GameToolExecutor construction failed",
            exc_info=True,
        )
        return None

    # Seed the speaker's own state so the model never has to
    # guess who is asking -- same trick guide plays with the
    # player's pushed context.
    if ctx:
        try:
            if ctx.get('zone_name'):
                executor.set_player_zone(ctx['zone_name'])
            executor.set_player_defaults(
                level=ctx.get('bot_level'),
                player_class=ctx.get('bot_class'),
                faction=ctx.get('faction'),
            )
        except Exception:
            logger.debug(
                "Executor context seeding failed",
                exc_info=True,
            )
    return executor


# ============================================================
# NATIVE TOOLS (no guide equivalent)
# ============================================================

NATIVE_TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'get_my_quest_log',
            'description': (
                'Return YOUR OWN current quest log: quest '
                'names, levels, and objective progress. Use '
                'this whenever you are asked what quests you '
                'are on, what you are working on, or where '
                'you are going next. Never answer those from '
                'memory -- you cannot know your quest log '
                'without calling this.'
            ),
            'parameters': {
                'type': 'object', 'properties': {},
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_dungeon_bosses',
            'description': (
                'List the bosses of a dungeon or raid IN ORDER, so '
                'the last one returned is the final boss. ALWAYS '
                'call this before naming any boss of an instance -- '
                'never answer from memory, you will confuse '
                'similarly-themed instances.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'dungeon_name': {
                        'type': 'string',
                        'description': (
                            "Dungeon or raid name, e.g. 'Deadmines', "
                            "'Shadowfang Keep', 'Blackwing Lair'."
                        ),
                    },
                },
                'required': ['dungeon_name'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'recall_memory',
            'description': (
                'Search your own memories of adventuring with '
                'this player for a topic. Use when asked '
                'whether you remember something, or about '
                'shared history. Returns only your own '
                'experiences with this player.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'topic': {
                        'type': 'string',
                        'description': (
                            'What to search for, e.g. "the '
                            'dungeon we cleared" or "when I '
                            'died".'
                        ),
                    },
                },
                'required': ['topic'],
            },
        },
    },
]

_QUEST_STATUS = {1: 'ready to turn in', 3: 'in progress',
                 5: 'failed'}


def _quest_log(config, ctx, limit=12):
    """The speaking bot's own quest log with progress.

    Nothing in mod-llm-chatter reads character_queststatus --
    quest events carry only a name and an id, which is why a
    bot asked "which quests do you have?" could only deflect.
    """
    from chatter_db import get_db_connection
    bot_guid = ctx.get('bot_guid')
    if not bot_guid:
        return 'no quest log available'

    cols_q = ', '.join(
        ['q.mobcount%d' % i for i in range(1, 5)] +
        ['q.itemcount%d' % i for i in range(1, 7)])
    cols_t = ', '.join(
        ['t.RequiredNpcOrGo%d' % i for i in range(1, 5)] +
        ['t.RequiredNpcOrGoCount%d' % i for i in range(1, 5)] +
        ['t.RequiredItemId%d' % i for i in range(1, 7)] +
        ['t.RequiredItemCount%d' % i for i in range(1, 7)] +
        ['t.ObjectiveText%d' % i for i in range(1, 5)])

    db = None
    try:
        db = get_db_connection(config)
        cur = db.cursor(dictionary=True)
        cur.execute(
            "SELECT q.quest, q.status, t.LogTitle,"
            " t.QuestLevel, %s, %s"
            " FROM acore_characters.character_queststatus q"
            " JOIN acore_world.quest_template t"
            "   ON t.ID = q.quest"
            " WHERE q.guid = %%s AND q.status IN (1, 3)"
            " ORDER BY q.status DESC, t.QuestLevel"
            " LIMIT %%s" % (cols_q, cols_t),
            (int(bot_guid), int(limit)),
        )
        rows = cur.fetchall()
        if not rows:
            return 'quest log is empty'

        # Resolve the names the objectives point at.
        npc_ids, go_ids, item_ids = set(), set(), set()
        for r in rows:
            for i in range(1, 5):
                v = r.get('RequiredNpcOrGo%d' % i) or 0
                if v > 0:
                    npc_ids.add(v)
                elif v < 0:
                    go_ids.add(-v)
            for i in range(1, 7):
                v = r.get('RequiredItemId%d' % i) or 0
                if v:
                    item_ids.add(v)

        def _names(table, ids):
            if not ids:
                return {}
            ph = ','.join(['%s'] * len(ids))
            cur.execute(
                "SELECT entry, name FROM acore_world.%s"
                " WHERE entry IN (%s)" % (table, ph),
                tuple(ids))
            return {r['entry']: r['name']
                    for r in cur.fetchall()}

        npcs = _names('creature_template', npc_ids)
        gos = _names('gameobject_template', go_ids)
        items = _names('item_template', item_ids)

        lines = []
        for r in rows:
            parts = []
            for i in range(1, 5):
                need = r.get('RequiredNpcOrGoCount%d' % i) or 0
                if not need:
                    continue
                ref = r.get('RequiredNpcOrGo%d' % i) or 0
                have = r.get('mobcount%d' % i) or 0
                name = (npcs.get(ref) if ref > 0
                        else gos.get(-ref)) or 'target'
                parts.append('%s %d/%d' % (name, have, need))
            for i in range(1, 7):
                need = r.get('RequiredItemCount%d' % i) or 0
                if not need:
                    continue
                ref = r.get('RequiredItemId%d' % i) or 0
                have = r.get('itemcount%d' % i) or 0
                name = items.get(ref) or 'item'
                parts.append('%s %d/%d' % (name, have, need))
            if not parts:
                for i in range(1, 5):
                    txt = (r.get('ObjectiveText%d' % i)
                           or '').strip()
                    if txt:
                        parts.append(txt)
            lines.append(
                '%s (level %s, %s)%s' % (
                    r.get('LogTitle') or 'unnamed quest',
                    r.get('QuestLevel'),
                    _QUEST_STATUS.get(r.get('status'),
                                      'in progress'),
                    (' -- ' + '; '.join(parts))
                    if parts else '',
                )
            )
        return '\n'.join(lines)
    except Exception:
        logger.error("quest log lookup failed", exc_info=True)
        return 'could not read quest log'
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


_dungeon_index = None


def _dungeon_map_index():
    """name -> map_id, built from DUNGEON_FLAVOR.

    map_dbc and dungeonencounter_dbc are both empty stubs on this
    world DB (0 and 16 rows), so there is no table mapping a
    dungeon's name to its map. DUNGEON_FLAVOR is keyed by map id
    with "Name: description" values, which is enough.
    """
    global _dungeon_index
    if _dungeon_index is not None:
        return _dungeon_index
    idx = {}
    try:
        from chatter_constants import DUNGEON_FLAVOR
        for map_id, text in DUNGEON_FLAVOR.items():
            name = str(text).split(':', 1)[0].strip()
            if not name:
                continue
            idx[_norm(name)] = (int(map_id), name)
            # "The Deadmines" should also answer to "Deadmines"
            bare = _norm(name)
            for article in ('the ', 'the'):
                if bare.startswith(article):
                    idx.setdefault(bare[len(article):], (int(map_id), name))
    except Exception:
        logger.debug("dungeon index unavailable", exc_info=True)
    _dungeon_index = idx
    return idx


def _norm(text):
    return ' '.join(str(text or '').lower().split())


def _resolve_dungeon(name):
    """Best-effort dungeon name -> (map_id, canonical name)."""
    idx = _dungeon_map_index()
    key = _norm(name)
    if key in idx:
        return idx[key]
    if key.startswith('the '):
        alt = key[4:]
        if alt in idx:
            return idx[alt]
    # substring, then fuzzy
    for k, v in idx.items():
        if key and (key in k or k in key):
            return v
    import difflib
    close = difflib.get_close_matches(key, list(idx), n=1, cutoff=0.7)
    return idx[close[0]] if close else (None, None)


def _dungeon_bosses(config, dungeon_name):
    """Ordered boss list for an instance.

    Uses instance_encounters, which is the encounter table the core
    itself credits kills against -- 628 rows here. The obvious
    alternatives do not work: creature_template.rank = 3 means WORLD
    boss, so a 5-man like Deadmines has none of them, and
    dungeonencounter_dbc is a 16-row stub.
    """
    from chatter_db import get_db_connection
    map_id, canonical = _resolve_dungeon(dungeon_name)
    if not map_id:
        return 'no dungeon by that name'
    db = None
    try:
        db = get_db_connection(config, database='acore_world')
        cur = db.cursor(dictionary=True)
        cur.execute(
            "SELECT ct.name"
            " FROM instance_encounters ie"
            " JOIN creature_template ct ON ct.entry = ie.creditEntry"
            " WHERE ie.creditEntry IN ("
            "   SELECT DISTINCT c.id FROM creature c WHERE c.map = %s)"
            " ORDER BY ie.entry",
            (int(map_id),),
        )
        names = [r['name'] for r in cur.fetchall()]
        if not names:
            return '%s: no encounters recorded' % canonical
        if len(names) == 1:
            return '%s: only boss is %s' % (canonical, names[0])
        return '%s bosses in order: %s. The FINAL boss is %s.' % (
            canonical, ', '.join(names[:-1]), names[-1])
    except Exception:
        logger.error("dungeon boss lookup failed", exc_info=True)
        return 'could not look that up'
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


def _recall(config, ctx, topic):
    """Search this bot's memories of this player.

    Prefers Hindsight (semantic, entity-aware) and falls back
    to a LIKE over llm_bot_memories.
    """
    try:
        from chatter_hindsight import recall_memories
        hits = recall_memories(config, ctx, topic)
        if hits:
            return '\n'.join(hits)
    except Exception:
        logger.debug("Hindsight recall failed", exc_info=True)

    from chatter_db import get_db_connection
    bot_guid = ctx.get('bot_guid')
    player_guid = ctx.get('player_guid')
    if not player_guid and ctx.get('player_name'):
        # Guild and proximity handlers know the name but not
        # the guid; memories are keyed on the guid.
        try:
            from chatter_db import get_character_info_by_name
            db0 = get_db_connection(config)
            try:
                info = get_character_info_by_name(
                    db0, ctx['player_name'])
                player_guid = (info or {}).get('guid')
            finally:
                db0.close()
        except Exception:
            logger.debug(
                "player guid lookup failed", exc_info=True)
    if not bot_guid or not player_guid:
        return 'no memories found'
    db = None
    try:
        db = get_db_connection(config)
        cur = db.cursor(dictionary=True)
        cur.execute(
            "SELECT memory FROM llm_bot_memories"
            " WHERE bot_guid = %s AND player_guid = %s"
            "   AND active = 1 AND memory LIKE %s"
            " ORDER BY created_at DESC LIMIT 5",
            (int(bot_guid), int(player_guid),
             '%%%s%%' % (topic or '')[:60]),
        )
        rows = cur.fetchall()
        if not rows:
            return 'no memories found about that'
        return '\n'.join(r['memory'] for r in rows)
    except Exception:
        logger.error("memory recall failed", exc_info=True)
        return 'no memories found'
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


# ============================================================
# ASSEMBLY
# ============================================================

def _sanitize_description(text):
    """Drop guide's link-marker instructions.

    Markers go first: the literal "[[npc:...]]" in the prose
    contains dots, which would otherwise chop sentence
    matching into fragments.
    """
    if not text:
        return ''
    cleaned = _ANY_MARKER_RE.sub('', text)
    cleaned = _MARKER_SENTENCE_RE.sub('', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    return cleaned or text


def select_tools(config, executor):
    """OpenAI-format tool schemas for the grounding call."""
    tools = list(NATIVE_TOOLS)
    guide = _load_guide(config) if executor is not None else None
    if guide is not None:
        raw = str(config.get(
            'LLMChatter.Tools.Allowed', DEFAULT_ALLOWED
        )).strip()
        if raw == '*':
            subset = list(guide.GAME_TOOLS)
        else:
            allowed = {n.strip() for n in raw.split(',')
                       if n.strip()}
            subset = [t for t in guide.GAME_TOOLS
                      if t.get('name') in allowed]
            missing = allowed - {t.get('name') for t in subset}
            if missing:
                logger.warning(
                    "Configured tools not present in guide's "
                    "catalogue: %s", sorted(missing),
                )
        # Guide ships convert_tools_to_openai_format, but it
        # lives in its bridge module which imports the whole
        # world. The mapping is three keys; do it here.
        tools += [{
            'type': 'function',
            'function': {
                'name': t['name'],
                'description': _sanitize_description(
                    t.get('description')),
                'parameters': t['input_schema'],
            },
        } for t in subset]
    return tools


def make_execute(config, executor, ctx):
    """Dispatch a tool call to the right backend."""
    def _execute(name, args):
        if name == 'get_my_quest_log':
            return _quest_log(config, ctx)
        if name == 'get_dungeon_bosses':
            return _dungeon_bosses(
                config, args.get('dungeon_name', ''))
        if name == 'recall_memory':
            return _recall(config, ctx, args.get('topic', ''))
        if executor is None:
            return 'lookup unavailable'
        result = executor.execute_tool(name, args or {})
        result = strip_link_markers(result or '')
        if len(result) > _MAX_RESULT_CHARS:
            result = result[:_MAX_RESULT_CHARS] + ' ...'
        return result
    return _execute


def ground(client, config, ctx, player_message,
           history=''):
    """Stage A: look things up before the bot speaks.

    Returns a <lookup_results> block to append to the normal
    prompt, or '' when nothing was looked up. Never raises.
    """
    if not int(config.get('LLMChatter.Tools.Enable', 0)):
        return ''
    if not player_message:
        return ''
    try:
        from chatter_llm import call_llm_with_tools

        executor = build_executor(config, ctx)
        tools = select_tools(config, executor)
        if not tools:
            return ''

        who = ctx.get('bot_name') or 'a companion'
        # The conversation so far. Without it a follow-up cannot be
        # resolved: "who's the final boss?" two lines after asking
        # about the Deadmines has no subject on its own, so nothing
        # gets looked up and the bot truthfully says it does not
        # know. The speaking prompt has always had this; the
        # grounding prompt was being built from the bare message.
        history_block = ''
        if history:
            history_block = (
                "Recent party chat, for resolving what is being "
                "referred to:\n%s\n\n" % str(history)[:1200]
            )
        prompt = (
            "You are %s, a level %s %s %s in World of "
            "Warcraft (3.3.5a), currently in %s.\n"
            "%s"
            "%s just said to you: \"%s\"\n\n"
            "If answering needs a fact you cannot know from "
            "this conversation -- a quest log, an item, an "
            "NPC, a quest chain, a place, or something you "
            "did together before -- call the tools that "
            "fetch it. The database is the source of truth "
            "for this server; your own recollection of World "
            "of Warcraft may be wrong or from a different "
            "expansion -- bosses and instances especially are "
            "easy to confuse. If the question refers back to "
            "something earlier in the chat above (\"the final "
            "boss\", \"how far is it\", \"what level\"), resolve "
            "what it means from that chat and look THAT up. "
            "easy to confuse, so look them up rather than "
            "trusting yourself. Prefer calling several tools at "
            "once over guessing. If no lookup is needed, reply "
            "with the single word NONE." % (
                who,
                ctx.get('bot_level') or '?',
                ctx.get('bot_race') or '',
                ctx.get('bot_class') or 'adventurer',
                ctx.get('zone_name') or 'Azeroth',
                history_block,
                ctx.get('player_name') or 'The player',
                str(player_message)[:400],
            )
        )

        results = call_llm_with_tools(
            client, prompt, config,
            tools=tools,
            execute=make_execute(config, executor, ctx),
            label='ground',
            max_rounds=int(config.get(
                'LLMChatter.Tools.MaxRounds', 2)),
            budget_seconds=float(config.get(
                'LLMChatter.Tools.BudgetSeconds', 45)),
        )
        if not results:
            return ''
        return (
            "\n<lookup_results>\n"
            "You looked these facts up from the game world just "
            "now. They are correct; your own recollection of "
            "World of Warcraft is not.\n%s\n"
            "ANSWER THE QUESTION WITH THESE FACTS. Be specific "
            "-- names, places, levels, numbers -- exactly as "
            "given. This overrides any length limit, creative "
            "twist or style note earlier in this prompt: those "
            "govern HOW you speak, never WHETHER you answer. "
            "Two sentences is fine if that is what it takes. A "
            "companion who knows the answer and gives a vague "
            "one instead is worse than useless.\n"
            "If these results genuinely do not cover what was "
            "asked, say so plainly -- but do not invent, and do "
            "not deflect with atmosphere.\n"
            "</lookup_results>\n"
            % '\n'.join('  - ' + r for r in results)
        )
    except Exception:
        logger.error("Grounding failed", exc_info=True)
        return ''


def ground_prompt(prompt, client, config, ctx,
                  player_message, history=''):
    """Append lookup results to an already-built prompt.

    PromptParts.__add__ keeps the system block intact and
    appends to the user half, so the JSON contract the
    parser depends on is unchanged.
    """
    block = ground(client, config, ctx, player_message,
                   history=history)
    return (prompt + block) if block else prompt


def ctx_from_bot(bot, player_name=None, player_guid=None,
                 zone_name=None):
    """Build a grounding context from a handler's bot dict."""
    return {
        'bot_guid': bot.get('guid'),
        'bot_name': bot.get('name'),
        'bot_level': bot.get('level'),
        'bot_class': bot.get('class'),
        'bot_race': bot.get('race'),
        'zone_name': zone_name,
        'player_name': player_name,
        'player_guid': player_guid,
    }
