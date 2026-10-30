"""en / ko / ja / cn strings for the MOSS-Image GUI, over ``gui.lines.strings``.

One module per language (``strings_{en,ko,ja,cn}``). Field labels/help: English
falls back to the request field's own ``help`` metadata (``library.moss.requests``);
ko / ja / cn cover the basic (non-advanced) fields, the rest fall back to English.
``FIELDS`` is keyed by request field name; ``FIELDS_GENERATE`` holds the
GenerateRequest fields whose name also exists on another request with a different
meaning.
"""

from __future__ import annotations

from gui.moss import strings_cn, strings_en, strings_ja, strings_ko

# Line-specific keys over the shared table in ``gui.lines.strings``.
UI: dict[str, dict[str, str]] = {
    "en": strings_en.UI,
    "ko": strings_ko.UI,
    "ja": strings_ja.UI,
    "cn": strings_cn.UI,
}
FIELDS: dict[str, dict[str, tuple[str, str]]] = {
    "ko": strings_ko.FIELDS,
    "ja": strings_ja.FIELDS,
    "cn": strings_cn.FIELDS,
}
FIELDS_GENERATE: dict[str, dict[str, tuple[str, str]]] = {
    "ko": strings_ko.FIELDS_GENERATE,
    "ja": strings_ja.FIELDS_GENERATE,
    "cn": strings_cn.FIELDS_GENERATE,
}
