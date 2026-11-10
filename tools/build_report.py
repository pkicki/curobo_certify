"""Build the self-contained planning report; optionally embed compact measured JSON."""
from pathlib import Path
import argparse
import html
import json
import re

ROOT = Path(__file__).resolve().parents[1]

def inline(text):
    text = html.escape(text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    return text

def markdown(text):
    # Small deterministic renderer for the repository-owned plan format.
    out, para, listing = [], [], False
    def flush():
        if para:
            out.append('<p>' + inline(' '.join(para)) + '</p>')
            para.clear()
    for line in text.splitlines():
        if line.startswith('- '):
            flush()
            if not listing:
                out.append('<ul>')
                listing = True
            out.append('<li>' + inline(line[2:]) + '</li>')
            continue
        if listing:
            out.append('</ul>')
            listing = False
        match = re.match(r'^(#{1,3}) (.*)$', line)
        if match:
            flush()
            n = len(match[1])
            out.append(f'<h{n}>' + inline(match[2]) + f'</h{n}>')
        elif not line.strip():
            flush()
        else:
            para.append(line)
    flush()
    if listing:
        out.append('</ul>')
    return '\n'.join(out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path)
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/timeopt_certification_campaign.html')
    args = parser.parse_args()
    plan = (ROOT / 'docs/CAMPAIGN_PLAN.md').read_text()
    prompt = (ROOT / 'docs/LUNA_PROMPT.md').read_text()
    results = json.loads(args.results.read_text()) if args.results else None
    template = (ROOT / 'tools/report_template.html').read_text()
    replacements = {
        '__PLAN_HTML__': markdown(plan), '__PROMPT_HTML__': markdown(prompt),
        '__MATH_JS__': (ROOT / 'tools/report_math.js').read_text(),
        '__CONTENT_JSON__': json.dumps({'plan': plan, 'prompt': prompt, 'results': results}, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c'),
    }
    # Replace each template marker once from the original template, never from injected text.
    output = re.sub('|'.join(map(re.escape, replacements)), lambda m: replacements[m[0]], template)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(output)
    print(f'Built {args.out} ({len(output.encode())} bytes)')

if __name__ == '__main__':
    main()
