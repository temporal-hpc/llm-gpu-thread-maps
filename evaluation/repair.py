"""Syntactic repair of the generated code.

Purpose: grade the mathematical reasoning of the model, not its formatting
discipline. No rule may alter an expression, a constant, an operator or the
control flow of the generated code.

  extraction   isolate the code from the surrounding prose (r1_extract)
  entry point  alias the mapping function when it is not named
               map_to_coordinates (r2_entrypoint)
  top level    drop module-level statements that are not imports,
               definitions or assignments (r3_strip_toplevel)

The names of math, itertools, functools and typing are made available by
the graders, so a missing import is not counted as a failure either.
"""
import ast, re, io, tokenize

FN = "map_to_coordinates"
# names that denote "the mapping function" and not a helper
ENTRY_RE = re.compile(
    r"(map|get|find|compute|calc|index|idx|nth|to)[_a-z0-9]*"
    r"(coord|coordinate|point|pos|position|xy|xyz|cell|block)", re.I)


def _parses(src):
    try:
        ast.parse(src); return True
    except SyntaxError:
        return False


def r1_extract(content):
    """Isolate the code. Prose never reaches the compiler."""
    fences = re.findall(r"```(?:[Pp]ython)?[ \t]*\r?\n(.*?)```", content, re.DOTALL)
    cands = [f.strip() for f in fences if "def " in f]
    if cands:
        # the largest fenced block that parses; if none parses, the largest one
        ok = [c for c in sorted(cands, key=len, reverse=True) if _parses(c)]
        return (ok[0] if ok else max(cands, key=len)), "fence"

    if _parses(content.strip()) and "def " in content:
        return content.strip(), "whole"

    # no fence: start at the first top-level 'def'/'import' and keep the
    # longest prefix that parses (cuts prose glued to the end)
    lines = content.splitlines()
    starts = [i for i, l in enumerate(lines) if re.match(r"(def |import |from )", l)]
    if not starts:
        return None, "no-code"
    s = starts[0]
    body = lines[s:]
    for end in range(len(body), 0, -1):
        chunk = "\n".join(body[:end]).rstrip()
        if "def " in chunk and _parses(chunk):
            return chunk, "carved"
    return None, "unparseable"


def r3_strip_toplevel(src):
    """Drop module-level statements other than import/def/class/assignment.

    Removes the test blocks some models append (print, input, loose loops,
    if __name__). Nothing inside a function is touched.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src, 0
    keep, dropped = [], 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                             ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
            keep.append(node)
        else:
            dropped += 1
    if not dropped:
        return src, 0
    tree.body = keep
    return ast.unparse(tree), dropped


def r2_entrypoint(src):
    """If map_to_coordinates does not exist, alias the mapping function.

    Candidate = module-level def with a single positional parameter that no
    other function of the module calls (that is, not a helper).
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src, None
    funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if any(f.name == FN for f in funcs):
        return src, None
    called = set()
    for f in funcs:
        for sub in ast.walk(f):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                called.add(sub.func.id)
    unary = [f for f in funcs
             if len(f.args.args) == 1 and not f.args.vararg and not f.args.kwonlyargs]
    pool = [f for f in unary if f.name not in called] or unary
    if not pool:
        return src, None
    named = [f for f in pool if ENTRY_RE.search(f.name)]
    pick = named[-1] if named else pool[-1]
    return src + "\n\n%s = %s\n" % (FN, pick.name), pick.name


def repair(content):
    log = {"extract": None, "dropped_toplevel": 0, "aliased_from": None}
    src, how = r1_extract(content)
    log["extract"] = how
    if src is None:
        return None, log
    src, log["dropped_toplevel"] = r3_strip_toplevel(src)
    src, log["aliased_from"] = r2_entrypoint(src)
    return src, log
