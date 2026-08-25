"""Era and lore guards for generated text.

The model's own Warcraft knowledge spans every expansion, while
this realm is frozen at Wrath of the Lich King (3.3.5a). Left
unchecked it reaches for whatever it remembers: a night elf from
Teldrassil was given a backstory set in Silvermoon -- the blood
elf capital -- fleeing "the Shattering" through "the Broken
Isles", two expansions that will not exist for years.

Two places need this, for different reasons:

  * Delivered chat lines are cheap and disposable, so an
    anachronism there is logged, not blocked -- a silent bot is
    worse than a slightly wrong one, and the log makes the rate
    measurable.

  * Backstories are generated ONCE and reused for the life of
    the character, so an error is permanent. Those are worth
    validating and regenerating, and the cost amortises to
    nothing.

Everything here is deliberately conservative. Terms that exist
in WotLK-era lore are NOT listed even when a later expansion
made them famous: Argus is the draenei homeworld, Draenor is
what Outland used to be, Gilneas and Kul Tiras are kingdoms on
the map, and Garrosh is already Warchief. Only names that
cannot be spoken in 3.3.5a are here.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Proper nouns that postdate Wrath of the Lich King.
_POST_WOTLK = [
    # Cataclysm
    'the shattering', 'cataclysm', 'vashj\'ir', 'deepholm',
    'twilight highlands', 'uldum', 'tol barad',
    # Mists of Pandaria
    'pandaria', 'pandaren', 'mogu', 'sha of', 'jade forest',
    'kun-lai', 'townlong', 'vale of eternal blossoms',
    'siege of orgrimmar',
    # Warlords of Draenor
    'iron horde', 'tanaan', 'frostfire ridge', 'gorgrond',
    'talador', 'ashran', 'draenor garrison',
    # Legion
    'broken isles', 'broken shore', 'suramar', 'val\'sharah',
    'stormheim', 'azsuna', 'highmountain', 'antorus',
    'class hall', 'artifact weapon', 'tomb of sargeras',
    # Battle for Azeroth
    'zandalar', 'azerite', 'nazjatar', 'mechagon',
    'heart of azeroth', 'fourth war', 'n\'zoth',
    # Shadowlands
    'shadowlands', 'the maw', 'torghast', 'bastion',
    'revendreth', 'ardenweald', 'maldraxxus', 'oribos',
    'the jailer', 'zovaal',
    # Dragonflight
    'dragon isles', 'valdrakken', 'dracthyr', 'thaldraszus',
    # The War Within
    'khaz algar', 'dornogal', 'nerub-ar', 'azj-kahet',
]

_ANACHRONISM_RE = re.compile(
    r'\b(' + '|'.join(re.escape(t) for t in _POST_WOTLK) + r')\b',
    re.IGNORECASE,
)

# Places strongly identified with one playable race. Used only to
# catch a character being given someone else's homeland; a bot can
# obviously travel, so this is applied to birthplace-shaped text
# (backstories), never to chat.
_RACE_HOMELANDS = {
    'human':     ['stormwind', 'elwynn', 'westfall', 'lordaeron',
                  'stromgarde', 'theramore', 'dalaran'],
    'dwarf':     ['ironforge', 'dun morogh', 'khaz modan',
                  'aerie peak'],
    'night elf': ['teldrassil', 'darnassus', 'ashenvale',
                  'moonglade', 'nighthaven', 'feralas',
                  'winterspring', 'darkshore'],
    'gnome':     ['gnomeregan', 'tinker town', 'dun morogh'],
    'draenei':   ['exodar', 'azuremyst', 'bloodmyst', 'argus',
                  'draenor', 'outland', 'shattrath'],
    'orc':       ['orgrimmar', 'durotar', 'draenor', 'outland',
                  'nagrand', 'blackrock'],
    'undead':    ['undercity', 'tirisfal', 'lordaeron',
                  'silverpine', 'deathknell'],
    'forsaken':  ['undercity', 'tirisfal', 'lordaeron',
                  'silverpine', 'deathknell'],
    'tauren':    ['thunder bluff', 'mulgore', 'bloodhoof',
                  'narache'],
    'troll':     ['durotar', 'echo isles', 'sen\'jin',
                  'stranglethorn', 'zul\'gurub'],
    'blood elf': ['silvermoon', 'eversong', 'quel\'thalas',
                  'ghostlands', 'tranquillien', 'sunstrider'],
}


def _norm(text):
    return ' '.join(str(text or '').lower().split())


def find_anachronisms(text):
    """Post-WotLK proper nouns present in text."""
    if not text:
        return []
    return sorted({m.group(1).lower()
                   for m in _ANACHRONISM_RE.finditer(str(text))})


def find_foreign_homelands(text, race):
    """Homelands belonging to a race that is not this one.

    Only reported when the character's OWN homeland is absent --
    "raised in Teldrassil, later studied in Silvermoon" is a
    perfectly good story, whereas "born in Silvermoon" for a
    night elf is the failure this exists to catch.
    """
    key = _norm(race)
    if key not in _RACE_HOMELANDS:
        return []
    body = _norm(text)
    if not body:
        return []
    if any(p in body for p in _RACE_HOMELANDS[key]):
        return []
    foreign = []
    for other, places in _RACE_HOMELANDS.items():
        if other == key:
            continue
        for place in places:
            # Shared homelands (Dun Morogh, Draenor, Lordaeron...)
            # belong to more than one race; never flag those.
            owners = [r for r, ps in _RACE_HOMELANDS.items()
                      if place in ps]
            if len(owners) > 1 or key in owners:
                continue
            if place in body:
                foreign.append(place)
    return sorted(set(foreign))


def audit_text(text, race=None, label=''):
    """Return a list of era/lore problems, empty when clean."""
    problems = []
    for term in find_anachronisms(text):
        problems.append('post-WotLK reference: %s' % term)
    if race:
        for place in find_foreign_homelands(text, race):
            problems.append(
                '%s homeland for a %s: %s'
                % ('foreign', race, place))
    if problems:
        logger.warning(
            "Lore audit failed (%s): %s | text=%.160s",
            label or 'text', '; '.join(problems), text,
        )
    return problems
