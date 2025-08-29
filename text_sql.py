import re
from jsonpath_ng.ext import parse as jp_parse
from jsonpath_ng.exceptions import JsonPathParserError

# --- Helpers to sanitize/repair LLM-emitted JSONPath strings ---
_ALLOWED_CHARS = r"[\w\.\@\$\[\]\?\(\)\=\!\<\>\s\'\"\*\-\/\:\,]+"  # conservative but generous

def _extract_jsonpath(expr: str) -> str:
    """
    From arbitrary LLM text, extract the first JSONPath-like substring
    that starts with '$' and contains only allowed characters.
    """
    # common prefixes the LLM might add
    expr = expr.strip()
    expr = re.sub(r"^JSONPath\s*:\s*", "", expr, flags=re.IGNORECASE)
    expr = re.sub(r"^(Path|Use this|Query)\s*:\s*", "", expr, flags=re.IGNORECASE)

    # strip code fences
    expr = re.sub(r"^```[a-zA-Z0-9]*\s*", "", expr)
    expr = re.sub(r"\s*```$", "", expr)

    # now extract the first $-prefixed chunk
    m = re.search(r"(\$" + _ALLOWED_CHARS + r")", expr)
    if m:
        return m.group(1).strip()
    return expr  # fallback

def _normalize_ops(expr: str) -> str:
    """
    Normalize operators/quotes to what jsonpath-ng expects.
    - Replace single '=' with '==', but keep '!=', '<=', '>=' intact.
    - Normalize fancy quotes to plain quotes.
    """
    # normalize quotes
    expr = expr.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    # fix single '=' inside filters: ...[@.x = 'y'] -> ...[@.x == 'y']
    # but don't touch '!=', '<=', '>=', '==' etc.
    def _eq_fix(match):
        left = match.group(1)
        right = match.group(2)
        return f"{left}=={right}"
    expr = re.sub(r"(@\.[A-Za-z_][A-Za-z0-9_]*\s*)=(\s*['\"][^'\"]*['\"])",
                  _eq_fix, expr)

    return expr

def _rename_common_fields(expr: str) -> str:
    """
    Map likely field name mistakes (author -> username).
    """
    # only rename inside @.field occurrences
    expr = re.sub(r"@\.author\b", "@.username", expr)
    return expr

def _fix_length_suffix(expr: str) -> str:
    """
    Replace trailing .length() with [*] for counting/selection purposes.
    """
    return expr.replace(".length()", "[*]")

def sanitize_jsonpath(expression: str) -> str:
    expr = expression.strip()
    expr = _extract_jsonpath(expr)
    expr = _fix_length_suffix(expr)
    expr = _rename_common_fields(expr)
    expr = _normalize_ops(expr)
    # jsonpath-ng is happier with double quotes; ensure string literals use them
    # Convert ['text'] -> ["text"] inside filters
    expr = re.sub(r"([=]=\s*)'([^']*)'", r'\1"\2"', expr)
    return expr

# --- Safer output processor using the sanitizer & retry strategy ---
def safe_output_processor(expression: str, data):
    """
    Execute JSONPath and return a list of match values.
    Attempts several sanitization passes before giving up.
    """
    candidates = []
    # 1) as-is
    candidates.append(expression)
    # 2) sanitized
    candidates.append(sanitize_jsonpath(expression))

    # If neither works, try a very defensive extraction: only keep from first '$'
    if '$' in expression:
        candidates.append(expression[expression.index('$'):].strip())

    last_exc = None
    for cand in candidates:
        try:
            matches = jp_parse(cand).find(data)
            return [m.value for m in matches]
        except JsonPathParserError as e:
            last_exc = e
            continue

    raise ValueError(f"Invalid JSON Path after attempted fixes: {expression}") from last_exc
