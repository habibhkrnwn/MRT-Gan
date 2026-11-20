"""Round-2 CPU check CB1 (light): who wrote the session-1 screening decisions (868 rows of
runs/18_review_mrtgan/review/review.csv). CPU only, reads existing files, writes agent_log_audit.json beside this script.

Inputs: the two Codex session logs of 3 Oct 2026 (outside the package, in ~/.codex/sessions/2026/10/03/), the final
review.csv, and the round-2 rating CSVs (hash only). The output is redacted: it holds no paths outside the package, no
user text, no e-mail address, and no blind-id-to-arm mapping. Run with `python -I`.

Checks:
  (a) per session: session id, first/last log timestamp, model identifier(s) and provider recorded in the log;
  (b) every `review_helper.py save` command (session 1) and every ratings.tsv heredoc (session 2; labels from the
      `ambiguous={...}` set of its write script) (CommandExecution items) is parsed. Each entry is
      [row_number, target_label, bits, reason] with bits in the order texture, shape, boundary, artifact (from the
      helper source in the same log). Parsed entries are compared with the final review.csv row by row;
  (c) whether any command in the logs prints the contents of review_key_PRIVATE.json (heuristic: an output that contains
      the key's '"arm":' field); directory listings that only show the file name are counted separately;
  (d) SHA-256 of the round-2 CSVs that were analysed earlier (all written by AI sessions per the session logs).
"""
import csv, hashlib, json, re
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
RV = HERE.parent
PKG = RV.parents[2]
R18 = PKG / '02_Hasil_Eksperimen/runs/18_review_mrtgan'
LOGDIR = Path.home() / '.codex/sessions/2026/10/03'
LOGS = sorted(LOGDIR.glob('rollout-2026-10-03T*.jsonl'))
REVIEW = R18 / 'review/review.csv'
ROUND2 = [RV / 'kappa/pack_heldout/review.csv', RV / 'kappa/pack_rater2/review.csv']
CRIT = ['texture_realistic', 'shape_ok', 'boundary_ok', 'artifact_free']


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wib(ts): return (datetime.fromisoformat(ts.replace('Z', '+00:00')) + timedelta(hours=7)).strftime('%Y-%m-%d %H:%M:%S WIB')


rows = list(csv.DictReader(open(REVIEW, newline='')))
assert len(rows) == 868
out = dict(inputs={'review_csv': dict(path=str(REVIEW.relative_to(PKG)), sha256=sha(REVIEW))},
           logs=[], round2_csv_sha256={str(p.relative_to(PKG)): sha(p) for p in ROUND2})
parsed_all = {}
for log in LOGS:
    lines = [json.loads(l) for l in open(log)]
    meta = next(d for d in lines if d['type'] == 'session_meta')['payload']
    models = sorted({m for d in lines for m in re.findall(r'"model":"([^"]+)"', json.dumps(d, separators=(',', ':')))})
    providers = sorted({m for d in lines for m in re.findall(r'"model_provider":"([^"]+)"', json.dumps(d, separators=(',', ':')))})
    saves, key_print, key_name_only = [], 0, 0
    # rows labelled ambiguous by the second session's write script (set literal in the same log)
    AMB = set()
    for d in lines:
        m = re.search(r"ambiguous=\{([0-9,]+)\}", json.dumps(d).replace('\\n', '\n'))
        if m:
            AMB = {int(x) for x in m.group(1).split(',')}
    for d in lines:
        p = d['payload']
        if d['type'] != 'event_msg' or p.get('type') != 'item_completed':
            continue
        item = p.get('item', {})
        cmd = item.get('command')
        if not cmd:
            continue
        c = cmd[-1]
        outtxt = str(item.get('aggregated_output', ''))
        if 'review_key_PRIVATE' in c or 'review_key_PRIVATE' in outtxt:
            if '"arm":' in outtxt or "'arm':" in outtxt:
                key_print += 1
            else:
                key_name_only += 1
        if 'review_helper.py save' in c:
            body = c.split("<<'JSON'\n", 1)[1].rsplit('\nJSON', 1)[0]
            saves.append((d['timestamp'], json.loads(body)))
        elif 'ratings.tsv <<' in c:  # second session: TSV heredocs "row<TAB>bits(y/n)<TAB>reason"
            body = c.split("<<'EOF'\n", 1)[1].split('\nEOF', 1)[0]
            ent = []
            for ln in body.split('\n'):
                m = re.match(r'^(\d+)\t([yn]{4})\t', ln)
                if m:
                    n = int(m.group(1)); bits = m.group(2).replace('y', '1').replace('n', '0')
                    ent.append([n, 'ambiguous' if n in AMB else 'unripe', bits, ''])
            saves.append((d['timestamp'], ent))
    n_entries, match, mism = 0, 0, []
    for ts, entries in saves:
        for number, label, bits, reason in entries:
            n_entries += 1
            r = rows[number - 1]
            ok = r['target_label'] == label and all(r[k] == ('yes' if b == '1' else 'no') for k, b in zip(CRIT, bits))
            match += ok
            if not ok:
                mism.append(number)
            parsed_all.setdefault(number, []).append(log.name[:30])
    rv = sorted({r['reviewer'] for r, in [(rows[n - 1],) for _, e in saves for n, *_ in e]}) if saves else []
    out['logs'].append(dict(
        file=log.name, sha256=sha(log), session_id=meta.get('id'), originator=meta.get('originator'),
        first_timestamp=wib(lines[0]['timestamp']), last_timestamp=wib(lines[-1]['timestamp']),
        model_identifiers=models, model_providers=providers,
        save_calls=len(saves), first_save=wib(saves[0][0]) if saves else None, last_save=wib(saves[-1][0]) if saves else None,
        entries_parsed=n_entries, entries_matching_final_csv=match, mismatching_rows=mism,
        reviewer_ids_of_saved_rows=rv,
        commands_printing_key_contents=key_print, commands_showing_key_file_name_only=key_name_only))
covered = sorted(parsed_all)
out['coverage'] = dict(rows_in_csv=868, rows_written_by_agent_save_calls=len(covered),
                       rows_saved_more_than_once=sum(len(v) > 1 for v in parsed_all.values()),
                       rows_not_covered=[i for i in range(1, 869) if i not in parsed_all])
out['reason_prefix_ai_all_rows'] = sum(r['reason'].startswith('Penilaian visual AI') for r in rows)
out['note'] = ('All rows covered by agent save calls and matching the final CSV means that every session-1 decision '
               '(label and four criteria) was written by the AI agent; no later edit is recorded in the file.')
json.dump(out, open(HERE / 'agent_log_audit.json', 'w'), indent=1)
print(json.dumps({k: out[k] for k in ('coverage', 'reason_prefix_ai_all_rows')}, indent=1))
for l in out['logs']:
    print({k: l[k] for k in ('file', 'model_identifiers', 'first_save', 'last_save', 'entries_parsed',
                             'entries_matching_final_csv', 'commands_printing_key_contents')})
