"""
Ad-hoc inspection of tokenizer failure cases: numbers, code, non-English text.
Not part of the training pipeline -- just prints token segmentations for probe strings.
"""
from nanochat.tokenizer import get_tokenizer

tok = get_tokenizer()

def show(label, text):
    ids = tok.encode(text)
    pieces = [tok.id_to_token(i) for i in ids]
    print(f"\n=== {label} ===")
    print(f"text:   {text!r}")
    print(f"tokens: {' | '.join(repr(p) for p in pieces)}")
    print(f"n_tokens={len(ids)}  n_chars={len(text)}  chars/token={len(text)/len(ids):.2f}")

# --- numbers ---
show("small integer", "42")
show("year-like", "2024")
show("long integer", "123456789")
show("decimal", "3.14159")
show("integer with commas", "1,234,567")
show("same number, +1 digit", "1234567890")  # compare boundary shift vs above

# --- source code ---
show("python snippet", "def foo(x, y):\n    return x + y")
show("indentation (4 spaces)", "    return x")
show("indentation (tab)", "\treturn x")
show("camelCase identifier", "getUserAccountBalance")
show("snake_case identifier", "get_user_account_balance")
show("operators", "x == y and a != b or c <= d")

# --- non-English text ---
show("Korean", "정직한 사실 위에")
show("Chinese", "你好,世界")
show("emoji", "great job! 🎉🚀")
show("accented Latin", "café résumé naïve")

# --- whitespace-sensitivity sanity check ---
show("no leading space", "cat")
show("leading space", " cat")
