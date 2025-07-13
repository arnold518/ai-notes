import re
from pathlib import Path

# Path to your markdown folder
root_dir = Path("")  # <-- update this

# Pattern that matches both `{width=50%}` and `{: width="50%"}`
pattern = re.compile(
    r"""^(?P<indent>[ \t]*)<center>\s*\n
(?P=indent)!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)
\{\s*(?:(?:width=(?P<width1>\d+(?:\.\d+)?)%)|  # match {width=50%}
(?:\:?\s*width="(?P<width2>\d+(?:\.\d+)?)%"))\s*\}\s*\n
(?P=indent)</center>\s*$""",
    re.MULTILINE | re.VERBOSE
)

def repl(match):
    indent = match.group("indent")
    alt = match.group("alt")
    src = match.group("src")
    width = match.group("width1") or match.group("width2")
    return f'{indent}![{alt}]({src}){{: .center style="width:{width}%;"}}\n'

# Apply to all markdown files
for md_file in root_dir.rglob("*.md"):
    content = md_file.read_text(encoding="utf-8")
    new_content = pattern.sub(repl, content)
    if content != new_content:
        print(f"Updated: {md_file}")
        md_file.write_text(new_content, encoding="utf-8")
