"""Write an offline, keyboard-driven review_tool.html next to a texture-study blind review.

    python tools/make_texture_review_tool.py --review <study>/p100/review

Reads review.csv (row order and any values already filled) and images/<blind_id>.png. The page
exports a CSV with exactly the columns of review.csv, so `texture_study.py select` can read it.
It never reads review_key_PRIVATE.json and shows no arm, parent or generator information.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

FIELDS = ['blind_id', 'target_label', 'texture_realistic', 'shape_ok', 'boundary_ok',
          'artifact_free', 'accepted', 'reviewer', 'reason']

PAGE = r'''<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review Tekstur Melon</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#15181d;--muted:#5d6675;--line:#dfe3ea;
  --accent:#2f6fed;--ok:#1f9d55;--no:#c8391f;--warn:#b3730a;--shadow:0 1px 3px rgba(0,0,0,.08)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#14161a;--panel:#1c1f25;--ink:#eceff4;--muted:#9aa4b2;--line:#2b303a;
  --accent:#6f9bff;--ok:#49c07d;--no:#ff7a5e;--warn:#e2a13a;--shadow:none}}
:root[data-theme="dark"]{--bg:#14161a;--panel:#1c1f25;--ink:#eceff4;--muted:#9aa4b2;--line:#2b303a;
  --accent:#6f9bff;--ok:#49c07d;--no:#ff7a5e;--warn:#e2a13a;--shadow:none}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:10px 16px;background:var(--panel);
  border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}
h1{font-size:16px;margin:0;font-weight:650}
.count{color:var(--muted);font-variant-numeric:tabular-nums}
.bar{flex:1 1 140px;height:8px;border-radius:99px;background:var(--line);overflow:hidden;min-width:100px}
.bar i{display:block;height:100%;background:var(--accent);width:0}
button,input[type=text]{font:inherit;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:7px 11px}
button{cursor:pointer}
button:hover:not(:disabled){border-color:var(--accent)}
button:disabled{opacity:.4;cursor:not-allowed}
input[type=text]{width:170px}
input.missing{border-color:var(--no)}
button.sel[data-kind="ripe"]{background:var(--warn);border-color:var(--warn);color:#10130f}
button.sel[data-kind="unripe"],button.sel[data-kind="yes"]{background:var(--ok);border-color:var(--ok);color:#06140c}
button.sel[data-kind="ambiguous"]{background:var(--muted);border-color:var(--muted);color:#0b0d11}
button.sel[data-kind="no"]{background:var(--no);border-color:var(--no);color:#170603}
main{display:grid;grid-template-columns:280px minmax(0,1fr);gap:16px;padding:16px;align-items:start}
#list{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:8px;
  max-height:calc(100vh - 100px);overflow:auto;box-shadow:var(--shadow)}
.filters{display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap}
.filters button{padding:4px 9px;font-size:13px}
.filters button.on{border-color:var(--accent);color:var(--accent)}
.thumbs{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
.thumb{position:relative;border:2px solid transparent;border-radius:8px;overflow:hidden;cursor:pointer;background:var(--line);aspect-ratio:1}
.thumb img{width:100%;height:100%;object-fit:cover;display:block}
.thumb.active{border-color:var(--accent)}
.thumb .tag{position:absolute;inset:auto 0 0 0;font-size:10px;text-align:center;background:rgba(0,0,0,.55);color:#fff}
.thumb.done.acc .tag{background:var(--ok);color:#06140c}
.thumb.done.rej .tag{background:var(--no);color:#170603}
.thumb.partial .tag{background:var(--warn);color:#10130f}
#stage{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;box-shadow:var(--shadow)}
.viewer{display:flex;gap:14px;flex-wrap:wrap;justify-content:center}
.frame{position:relative;width:min(512px,100%);aspect-ratio:1;background:#000;border-radius:10px;overflow:hidden;cursor:crosshair}
.frame img{display:block;width:100%;height:100%}
.lens{width:min(256px,100%);aspect-ratio:1;border-radius:10px;border:1px solid var(--line);background:#000 no-repeat;
  image-rendering:pixelated;background-size:2048px 2048px}
.lenswrap{display:flex;flex-direction:column;gap:6px;align-items:center}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:12px}
.label{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);width:100%;margin-bottom:-4px}
.crit{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px 18px;margin-top:12px}
.crit .row{margin-top:0}
textarea{width:100%;min-height:52px;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:9px;padding:8px;font:inherit;resize:vertical}
.hint{color:var(--muted);font-size:13px}
.warn{color:var(--no);font-size:13px}
kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:5px;padding:0 5px;font-size:12px}
.id{font-variant-numeric:tabular-nums;font-weight:650}
@media (max-width:820px){main{grid-template-columns:1fr}#list{max-height:220px;order:2}.crit{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>Review tekstur (buta)</h1>
  <span class="count" id="count">0 / 0</span>
  <span class="bar"><i id="bar"></i></span>
  <input type="text" id="reviewer" placeholder="nama reviewer (wajib)">
  <button id="export">Unduh review.csv</button>
  <label class="hint">Muat CSV <input type="file" id="import" accept=".csv" style="max-width:150px"></label>
</header>
<main>
  <section id="list">
    <div class="filters">
      <button data-filter="all" class="on">semua</button>
      <button data-filter="todo">belum</button>
      <button data-filter="acc">accepted</button>
      <button data-filter="rej">ditolak</button>
    </div>
    <div class="thumbs" id="thumbs"></div>
  </section>
  <section id="stage">
    <div class="row" style="margin-top:0"><span class="id" id="rid">—</span><span class="hint" id="pos"></span></div>
    <div class="viewer">
      <div class="frame" id="frame"><img id="shot" alt="kandidat"></div>
      <div class="lenswrap"><div class="lens" id="lens"></div>
        <span class="hint">Lensa 4× piksel asli. Arahkan kursor ke gambar.</span></div>
    </div>
    <div class="row"><span class="label">Kelas kematangan yang terlihat</span>
      <button data-field="target_label" data-kind="ripe" data-value="ripe">ripe <kbd>1</kbd></button>
      <button data-field="target_label" data-kind="unripe" data-value="unripe">unripe <kbd>2</kbd></button>
      <button data-field="target_label" data-kind="ambiguous" data-value="ambiguous">ambiguous <kbd>3</kbd></button>
    </div>
    <div class="crit">
      <div class="row"><span class="label">Tekstur realistis</span>
        <button data-field="texture_realistic" data-kind="yes" data-value="yes">yes <kbd>q</kbd></button>
        <button data-field="texture_realistic" data-kind="no" data-value="no">no <kbd>a</kbd></button></div>
      <div class="row"><span class="label">Bentuk buah wajar</span>
        <button data-field="shape_ok" data-kind="yes" data-value="yes">yes <kbd>w</kbd></button>
        <button data-field="shape_ok" data-kind="no" data-value="no">no <kbd>s</kbd></button></div>
      <div class="row"><span class="label">Batas buah–latar rapi</span>
        <button data-field="boundary_ok" data-kind="yes" data-value="yes">yes <kbd>e</kbd></button>
        <button data-field="boundary_ok" data-kind="no" data-value="no">no <kbd>d</kbd></button></div>
      <div class="row"><span class="label">Bebas artefak (grid, bercak, noise warna)</span>
        <button data-field="artifact_free" data-kind="yes" data-value="yes">yes <kbd>r</kbd></button>
        <button data-field="artifact_free" data-kind="no" data-value="no">no <kbd>f</kbd></button></div>
    </div>
    <div class="row"><span class="hint"><kbd>Enter</kbd> = keempat kriteria yes</span></div>
    <div class="row"><span class="label">Keputusan akhir (otomatis dari label + kriteria)</span>
      <button data-field="accepted" data-kind="yes" data-value="yes" id="accYes">accepted <kbd>y</kbd></button>
      <button data-field="accepted" data-kind="no" data-value="no">ditolak <kbd>n</kbd></button>
      <span class="warn" id="accWarn"></span>
    </div>
    <div class="row"><span class="label">Alasan (opsional)</span>
      <textarea id="reason" placeholder="mis. grid halus di permukaan, highlight kabur, batas bergerigi"></textarea></div>
    <div class="row">
      <button id="prev">← sebelumnya <kbd>←</kbd></button>
      <button id="next">berikutnya → <kbd>→</kbd></button>
      <span class="hint" id="owner"></span>
      <span class="hint">Pindah otomatis setelah label, empat kriteria dan keputusan terisi. Isian tersimpan di browser; unduh CSV secara berkala.</span>
    </div>
  </section>
</main>
<script>
const ITEMS=__ITEMS__;
const SEED=__SEED__;
const KEY='melon-texture-review-'+__TAG__;
const CRIT=['texture_realistic','shape_ok','boundary_ok','artifact_free'];
const FIELDS=['target_label',...CRIT,'accepted'];
let state={},cur=0,filter='all',reviewer='';
// review.csv values (SEED) are the base; answers already given in this browser win per item.
state=Object.assign({},SEED);reviewer='';
try{const local=JSON.parse(localStorage.getItem(KEY)||'{}');
  Object.keys(local).forEach(id=>{const r=local[id]||{};if(FIELDS.some(f=>r[f]))state[id]=r;});
  reviewer=localStorage.getItem(KEY+'-reviewer')||reviewer;}catch(e){}
const $=id=>document.getElementById(id);
function save(){try{localStorage.setItem(KEY,JSON.stringify(state));localStorage.setItem(KEY+'-reviewer',reviewer);}catch(e){}}
function rec(id){return state[id]||(state[id]={target_label:'',texture_realistic:'',shape_ok:'',boundary_ok:'',artifact_free:'',accepted:'',reviewer:'',reason:''});}
function done(r){return FIELDS.every(f=>r[f]);}
function any(r){return FIELDS.some(f=>r[f]);}
function canAccept(r){return r.target_label==='unripe'&&CRIT.every(f=>r[f]==='yes');}
function visible(){return ITEMS.filter(id=>{const r=state[id]||{};
  return filter==='all'||(filter==='todo'&&!done(r))||(filter==='acc'&&r.accepted==='yes')||(filter==='rej'&&r.accepted==='no');});}

function paintList(){
  const box=$('thumbs');box.innerHTML='';
  visible().forEach(id=>{
    const r=state[id]||{},i=ITEMS.indexOf(id),cell=document.createElement('div');
    cell.className='thumb'+(i===cur?' active':'')+(done(r)?' done '+(r.accepted==='yes'?'acc':'rej'):(any(r)?' partial':''));
    cell.innerHTML=`<img loading="lazy" src="images/${id}.png" alt=""><span class="tag">${i+1}</span>`;
    cell.onclick=()=>{cur=i;paint();};
    box.appendChild(cell);
  });
}
function paint(){
  const id=ITEMS[cur],r=rec(id);
  $('shot').src=`images/${id}.png`;
  $('lens').style.backgroundImage=`url(images/${id}.png)`;
  $('rid').textContent=id;$('pos').textContent=`${cur+1} dari ${ITEMS.length}`;
  $('reason').value=r.reason||'';
  $('owner').textContent=r.reviewer?('reviewer baris ini: '+r.reviewer):'';
  document.querySelectorAll('button[data-field]').forEach(b=>b.classList.toggle('sel',r[b.dataset.field]===b.dataset.value));
  $('accYes').disabled=!canAccept(r);
  $('accWarn').textContent=canAccept(r)?'':'accepted hanya bila label unripe dan keempat kriteria yes';
  const finished=ITEMS.filter(x=>done(state[x]||{})).length,acc=ITEMS.filter(x=>(state[x]||{}).accepted==='yes').length;
  $('count').textContent=`${finished} / ${ITEMS.length} selesai · ${acc} accepted`;
  $('bar').style.width=(100*finished/ITEMS.length)+'%';
  $('reviewer').value=reviewer;$('reviewer').classList.toggle('missing',!reviewer.trim());
  paintList();
  const active=document.querySelector('.thumb.active');if(active)active.scrollIntoView({block:'nearest'});
}
function set(field,value,toggle=true){
  const id=ITEMS[cur],r=rec(id),was=done(r);
  r[field]=(toggle&&r[field]===value)?'':value;
  if(reviewer.trim())r.reviewer=reviewer.trim();  // whoever changes a row owns it
  // Same rule as the converted petani_nilai rows: accepted follows label + four criteria.
  if(r.target_label&&CRIT.every(f=>r[f]))r.accepted=canAccept(r)?'yes':'no';else if(field!=='accepted')r.accepted='';
  save();paint();
  if(!was&&done(r)&&cur<ITEMS.length-1)setTimeout(()=>{cur++;paint();},220);
}
function allYes(){const r=rec(ITEMS[cur]);CRIT.forEach(f=>r[f]='yes');set('texture_realistic','yes',false);}
document.querySelectorAll('button[data-field]').forEach(b=>b.onclick=()=>set(b.dataset.field,b.dataset.value));
document.querySelectorAll('.filters button').forEach(b=>b.onclick=()=>{filter=b.dataset.filter;
  document.querySelectorAll('.filters button').forEach(x=>x.classList.toggle('on',x===b));paintList();});
$('reason').oninput=e=>{const r=rec(ITEMS[cur]);r.reason=e.target.value;if(reviewer.trim())r.reviewer=reviewer.trim();save();};
$('reviewer').oninput=e=>{reviewer=e.target.value;save();$('reviewer').classList.toggle('missing',!reviewer.trim());};
$('prev').onclick=()=>{if(cur>0){cur--;paint();}};
$('next').onclick=()=>{if(cur<ITEMS.length-1){cur++;paint();}};
$('frame').onmousemove=e=>{
  const b=$('frame').getBoundingClientRect(),lens=$('lens'),size=lens.clientWidth,zoom=4*b.width;
  const x=(e.clientX-b.left)/b.width,y=(e.clientY-b.top)/b.height;
  lens.style.backgroundSize=`${zoom}px ${zoom}px`;
  lens.style.backgroundPosition=`${-(x*zoom-size/2)}px ${-(y*zoom-size/2)}px`;
};
document.onkeydown=e=>{
  if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT')return;
  const k=e.key.toLowerCase();
  const map={'1':['target_label','ripe'],'2':['target_label','unripe'],'3':['target_label','ambiguous'],
    'q':['texture_realistic','yes'],'a':['texture_realistic','no'],'w':['shape_ok','yes'],'s':['shape_ok','no'],
    'e':['boundary_ok','yes'],'d':['boundary_ok','no'],'r':['artifact_free','yes'],'f':['artifact_free','no'],
    'y':['accepted','yes'],'n':['accepted','no']};
  if(map[k]){e.preventDefault();if(k==='y'&&!canAccept(rec(ITEMS[cur])))return;set(...map[k]);return;}
  if(e.key==='Enter'){e.preventDefault();allYes();return;}
  if(e.key==='ArrowLeft'&&cur>0){cur--;paint();}
  if(e.key==='ArrowRight'&&cur<ITEMS.length-1){cur++;paint();}
};
function cell(t){t=String(t==null?'':t);return /[",\n]/.test(t)?'"'+t.replace(/"/g,'""')+'"':t;}
$('export').onclick=()=>{
  if(ITEMS.some(id=>{const r=state[id]||{};return any(r)&&!(r.reviewer||reviewer.trim());})){alert('Isi nama reviewer dulu.');$('reviewer').focus();return;}
  const cols=['blind_id',...FIELDS,'reviewer','reason'],rows=[cols.join(',')];
  ITEMS.forEach(id=>{const r=state[id]||{};
    rows.push([id,...FIELDS.map(f=>r[f]||''),any(r)?cell(r.reviewer||reviewer.trim()):'',cell(r.reason||'')].join(','));});
  const blob=new Blob([rows.join('\n')+'\n'],{type:'text/csv'}),a=document.createElement('a');
  a.href=URL.createObjectURL(blob);a.download='review.csv';a.click();URL.revokeObjectURL(a.href);
};
function parseCSV(text){const out=[];let row=[],c='',q=false;
  for(let i=0;i<text.length;i++){const ch=text[i];
    if(q){if(ch==='"'&&text[i+1]==='"'){c+='"';i++;}else if(ch==='"')q=false;else c+=ch;}
    else if(ch==='"')q=true;else if(ch===','){row.push(c);c='';}
    else if(ch==='\n'||ch==='\r'){if(ch==='\r'&&text[i+1]==='\n')i++;row.push(c);c='';if(row.some(x=>x))out.push(row);row=[];}
    else c+=ch;}
  row.push(c);if(row.some(x=>x))out.push(row);return out;}
$('import').onchange=e=>{
  const file=e.target.files[0];if(!file)return;const reader=new FileReader();
  reader.onload=()=>{const rows=parseCSV(reader.result),head=rows.shift();let n=0;
    rows.forEach(v=>{const o={};head.forEach((h,i)=>o[h]=v[i]||'');
      if(!ITEMS.includes(o.blind_id))return;n++;
      state[o.blind_id]={target_label:o.target_label,texture_realistic:o.texture_realistic,shape_ok:o.shape_ok,
        boundary_ok:o.boundary_ok,artifact_free:o.artifact_free,accepted:o.accepted,reviewer:o.reviewer,reason:o.reason};});
    save();paint();alert(n+' baris dimuat');};
  reader.readAsText(file);
};
{const first=ITEMS.findIndex(id=>!done(state[id]||{}));cur=first<0?0:first;}
paint();
</script>
</body>
</html>
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--review', required=True, help='public review folder with review.csv and images/')
    args = ap.parse_args()
    review = Path(args.review)
    with (review/'review.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or list(rows[0]) != FIELDS:
        raise ValueError(f'review.csv columns must be {FIELDS}')
    ids = [r['blind_id'] for r in rows]
    missing = [i for i in ids if not (review/'images'/f'{i}.png').exists()]
    if missing or len(set(ids)) != len(ids):
        raise ValueError(f'missing images {missing[:5]} or duplicate ids')
    if (review/'review_key_PRIVATE.json').exists():
        raise ValueError('private key must not be inside the public review folder')
    seed = {r['blind_id']: {k: r[k] for k in FIELDS[1:]} for r in rows if any(r[k] for k in FIELDS[1:])}
    tag = hashlib.sha256('\n'.join(ids).encode()).hexdigest()[:12]
    page = (PAGE.replace('__ITEMS__', json.dumps(ids)).replace('__SEED__', json.dumps(seed))
            .replace('__TAG__', json.dumps(tag)))
    (review/'review_tool.html').write_text(page, encoding='utf-8')
    print(review/'review_tool.html', len(ids), 'items')


if __name__ == '__main__':
    main()
