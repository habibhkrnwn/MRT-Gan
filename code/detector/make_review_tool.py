"""Build an offline blind-review page for a generated bundle.

    python make_review_tool.py <bundle_dir>

The page keeps the review blind: it shows the candidate image, an optional outline of the
inherited mask, and nothing else. No method, no parent, no target class reaches the browser.
Masks are copied under review IDs so the filename leaks nothing either.
"""
import json, shutil, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
FIELDS=['review_id','image','label_class','annotation_ok','notes']


def build(bundle):
    bundle=Path(bundle);review=bundle/'review'
    key=json.loads((review/'key.json').read_text())
    records={r['candidate_id']:r for r in json.loads((bundle/'records.json').read_text())}
    masks=review/'blind_masks';masks.mkdir(exist_ok=True)
    items=[]
    for review_id in sorted(key):
        record=records[key[review_id]]
        shutil.copyfile(bundle/record['mask'],masks/f'{review_id}.png')
        items.append(dict(id=review_id,image=f'blind/{review_id}.png',
                          mask=f'blind_masks/{review_id}.png',bbox=record['bbox']))
    page=TEMPLATE.replace('__ITEMS__',json.dumps(items,separators=(',',':')))
    (review/'review_tool.html').write_text(page)
    print(f'wrote {review/"review_tool.html"} with {len(items)} candidates')
    print(f'masks copied under review ids into {masks}')
    return review/'review_tool.html'


TEMPLATE=r"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review Kandidat</title>
<style>
:root{
  --bg:#f6f7f9; --panel:#ffffff; --ink:#15181d; --muted:#5d6675; --line:#dfe3ea;
  --accent:#2f6fed; --ok:#1f9d55; --no:#c8391f; --warn:#b3730a; --shadow:0 1px 3px rgba(0,0,0,.08);
}
:root:not([data-theme="light"]){ @media (prefers-color-scheme: dark){
  :root{ --bg:#14161a; --panel:#1c1f25; --ink:#eceff4; --muted:#9aa4b2; --line:#2b303a;
         --accent:#6f9bff; --ok:#49c07d; --no:#ff7a5e; --warn:#e2a13a; --shadow:none; }
}}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#14161a; --panel:#1c1f25; --ink:#eceff4; --muted:#9aa4b2; --line:#2b303a;
  --accent:#6f9bff; --ok:#49c07d; --no:#ff7a5e; --warn:#e2a13a; --shadow:none; }}
:root[data-theme="dark"]{ --bg:#14161a; --panel:#1c1f25; --ink:#eceff4; --muted:#9aa4b2;
  --line:#2b303a; --accent:#6f9bff; --ok:#49c07d; --no:#ff7a5e; --warn:#e2a13a; --shadow:none; }
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
header{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:12px 16px;
       background:var(--panel);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}
h1{font-size:16px;margin:0;font-weight:650}
.count{color:var(--muted);font-variant-numeric:tabular-nums}
.bar{flex:1 1 160px;height:8px;border-radius:99px;background:var(--line);overflow:hidden;min-width:120px}
.bar i{display:block;height:100%;background:var(--accent);width:0}
button{font:inherit;color:var(--ink);background:var(--panel);border:1px solid var(--line);
       border-radius:9px;padding:8px 12px;cursor:pointer}
button:hover{border-color:var(--accent)}
button.sel[data-kind="ripe"]{background:var(--warn);border-color:var(--warn);color:#10130f}
button.sel[data-kind="unripe"]{background:var(--ok);border-color:var(--ok);color:#06140c}
button.sel[data-kind="unsure"]{background:var(--muted);border-color:var(--muted);color:#0b0d11}
button.sel[data-kind="yes"]{background:var(--ok);border-color:var(--ok);color:#06140c}
button.sel[data-kind="no"]{background:var(--no);border-color:var(--no);color:#170603}
main{display:grid;grid-template-columns:260px minmax(0,1fr);gap:16px;padding:16px;align-items:start}
#list{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:8px;
      max-height:calc(100vh - 110px);overflow:auto;box-shadow:var(--shadow)}
.thumbs{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
.thumb{position:relative;border:2px solid transparent;border-radius:8px;overflow:hidden;
       cursor:pointer;background:var(--line);aspect-ratio:1}
.thumb img{width:100%;height:100%;object-fit:cover;display:block}
.thumb.active{border-color:var(--accent)}
.thumb .tag{position:absolute;inset:auto 0 0 0;font-size:10px;text-align:center;
            background:rgba(0,0,0,.55);color:#fff;padding:1px 0}
.thumb.done .tag{background:var(--ok);color:#06140c}
.thumb.partial .tag{background:var(--warn);color:#10130f}
#stage{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;
       box-shadow:var(--shadow)}
.frame{position:relative;width:min(512px,100%);margin:0 auto;background:#000;border-radius:10px;overflow:hidden}
.frame img{display:block;width:100%}
.frame .overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.frame .mask{position:absolute;inset:0;width:100%;opacity:.4;mix-blend-mode:screen;
             pointer-events:none;display:none}
.frame.fallback.show-mask .mask{display:block}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:14px}
.label{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);
       width:100%;margin-bottom:-4px}
textarea{width:100%;min-height:56px;background:var(--bg);color:var(--ink);border:1px solid var(--line);
         border-radius:9px;padding:8px;font:inherit;resize:vertical}
.hint{color:var(--muted);font-size:13px}
kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:5px;padding:0 5px;font-size:12px}
.id{font-variant-numeric:tabular-nums;font-weight:650}
@media (max-width:780px){ main{grid-template-columns:1fr;padding:16px}
  #list{max-height:200px;order:2} }
</style>
</head>
<body>
<header>
  <h1>Review kandidat (buta)</h1>
  <span class="count" id="count">0 / 0</span>
  <span class="bar"><i id="bar"></i></span>
  <button id="export">Unduh CSV</button>
  <label class="hint">Muat CSV <input type="file" id="import" accept=".csv" style="max-width:150px"></label>
</header>
<main>
  <section id="list"><div class="thumbs" id="thumbs"></div></section>
  <section id="stage">
    <div class="row"><span class="id" id="rid">—</span><span class="hint" id="pos"></span></div>
    <div class="frame" id="frame">
      <img id="shot" alt="kandidat">
      <img class="mask" id="maskimg" alt="">
      <canvas class="overlay" id="overlay" width="512" height="512"></canvas>
    </div>
    <div class="row">
      <button id="mask">Garis mask: on <kbd>m</kbd></button>
      <button id="box">Kotak bbox: on <kbd>b</kbd></button>
      <span class="hint">Garis kuning = mask warisan, kotak biru putus-putus = bbox anotasi.
      Matikan keduanya untuk melihat citra polos.</span>
    </div>
    <div class="row"><span class="label">Kelas kematangan yang terlihat</span>
      <button data-field="label_class" data-kind="ripe" data-value="ripe">ripe <kbd>1</kbd></button>
      <button data-field="label_class" data-kind="unripe" data-value="unripe">unripe <kbd>2</kbd></button>
      <button data-field="label_class" data-kind="unsure" data-value="unsure">unsure <kbd>3</kbd></button>
    </div>
    <div class="row"><span class="label">Anotasi warisan masih pas?</span>
      <button data-field="annotation_ok" data-kind="yes" data-value="yes">yes <kbd>y</kbd></button>
      <button data-field="annotation_ok" data-kind="no" data-value="no">no <kbd>n</kbd></button>
    </div>
    <div class="row"><span class="label">Catatan (opsional)</span>
      <textarea id="notes" placeholder="artefak, buah ganda, batas mask meleset"></textarea></div>
    <div class="row">
      <button id="prev">← sebelumnya <kbd>←</kbd></button>
      <button id="next">berikutnya → <kbd>→</kbd></button>
      <span class="hint">Pindah otomatis setelah dua pilihan terisi. Pilihan dapat diubah kapan saja.</span>
    </div>
  </section>
</main>
<script>
const ITEMS=__ITEMS__;
const KEY='melon-blind-review-v1';
let state={},cur=0,showMask=true,showBox=true;
try{ state=JSON.parse(localStorage.getItem(KEY)||'{}'); }catch(e){ state={}; }
const $=id=>document.getElementById(id);

function save(){ try{ localStorage.setItem(KEY,JSON.stringify(state)); }catch(e){} }
function rec(i){ return state[ITEMS[i].id]||(state[ITEMS[i].id]={label_class:'',annotation_ok:'',notes:''}); }
function done(r){ return r.label_class && r.annotation_ok; }

function paintList(){
  $('thumbs').innerHTML='';
  ITEMS.forEach((item,i)=>{
    const r=state[item.id]||{};
    const cell=document.createElement('div');
    cell.className='thumb'+(i===cur?' active':'')+(done(r)?' done':(r.label_class||r.annotation_ok?' partial':''));
    cell.innerHTML=`<img loading="lazy" src="${item.image}" alt=""><span class="tag">${item.id.replace('R','')}</span>`;
    cell.onclick=()=>{ cur=i; paint(); };
    $('thumbs').appendChild(cell);
  });
}

const maskImage=new Image();
maskImage.onload=()=>drawOverlay();

function drawOverlay(){
  const item=ITEMS[cur], ctx=$('overlay').getContext('2d');
  ctx.clearRect(0,0,512,512);
  if(showMask && maskImage.complete && maskImage.naturalWidth){
    const off=document.createElement('canvas'); off.width=off.height=512;
    const octx=off.getContext('2d',{willReadFrequently:true});
    octx.drawImage(maskImage,0,0,512,512);
    let src;
    try{ src=octx.getImageData(0,0,512,512).data; }
    catch(err){ $('frame').classList.add('fallback','show-mask'); src=null; }
    if(src){
      $('frame').classList.remove('fallback');
      const out=ctx.createImageData(512,512), px=out.data;
      const on=(x,y)=> x<0||y<0||x>511||y>511 ? false : src[((y*512+x)<<2)]>127;
      for(let y=0;y<512;y++) for(let x=0;x<512;x++){
        if(!on(x,y)) continue;
        if(on(x-1,y)&&on(x+1,y)&&on(x,y-1)&&on(x,y+1)) continue;   // interior pixel
        for(let dy=-1;dy<=1;dy++) for(let dx=-1;dx<=1;dx++){       // 3px wide outline
          const nx=x+dx, ny=y+dy;
          if(nx<0||ny<0||nx>511||ny>511) continue;
          const j=((ny*512+nx)<<2);
          px[j]=255; px[j+1]=214; px[j+2]=64; px[j+3]=255;
        }
      }
      ctx.putImageData(out,0,0);
    }
  } else {
    $('frame').classList.remove('show-mask');
  }
  if(showBox){
    const [x,y,w,h]=item.bbox;
    ctx.save(); ctx.strokeStyle='#4f8bff'; ctx.lineWidth=3; ctx.setLineDash([10,7]);
    ctx.strokeRect(x+1.5,y+1.5,w-3,h-3); ctx.restore();
  }
}

function paint(){
  const item=ITEMS[cur], r=rec(cur);
  $('shot').src=item.image;
  $('maskimg').src=item.mask;
  if(maskImage.src.indexOf(item.mask)<0) maskImage.src=item.mask; else drawOverlay();
  $('rid').textContent=item.id; $('pos').textContent=`${cur+1} dari ${ITEMS.length}`;
  $('notes').value=r.notes||'';
  document.querySelectorAll('button[data-field]').forEach(b=>{
    b.classList.toggle('sel', r[b.dataset.field]===b.dataset.value); });
  const finished=ITEMS.filter(it=>done(state[it.id]||{})).length;
  $('count').textContent=`${finished} / ${ITEMS.length} selesai`;
  $('bar').style.width=(100*finished/ITEMS.length)+'%';
  paintList();
  const active=document.querySelector('.thumb.active');
  if(active) active.scrollIntoView({block:'nearest'});
}

function set(field,value){
  const r=rec(cur), was=done(r);
  r[field]=(r[field]===value)?'':value;
  save();
  const nowDone=done(r);
  paint();
  if(!was && nowDone && cur<ITEMS.length-1){ setTimeout(()=>{ cur++; paint(); },220); }
}

document.querySelectorAll('button[data-field]').forEach(b=>{
  b.onclick=()=>set(b.dataset.field,b.dataset.value); });
$('notes').oninput=e=>{ rec(cur).notes=e.target.value; save(); };
$('prev').onclick=()=>{ if(cur>0){cur--;paint();} };
$('next').onclick=()=>{ if(cur<ITEMS.length-1){cur++;paint();} };
$('mask').onclick=()=>{ showMask=!showMask;
  $('mask').innerHTML='Garis mask: '+(showMask?'on':'off')+' <kbd>m</kbd>'; drawOverlay(); };
$('box').onclick=()=>{ showBox=!showBox;
  $('box').innerHTML='Kotak bbox: '+(showBox?'on':'off')+' <kbd>b</kbd>'; drawOverlay(); };

document.onkeydown=e=>{
  if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT') return;
  const map={'1':['label_class','ripe'],'2':['label_class','unripe'],'3':['label_class','unsure'],
             'y':['annotation_ok','yes'],'n':['annotation_ok','no']};
  const hit=map[e.key.toLowerCase()];
  if(hit){ e.preventDefault(); set(hit[0],hit[1]); return; }
  if(e.key==='ArrowLeft'&&cur>0){ cur--; paint(); }
  if(e.key==='ArrowRight'&&cur<ITEMS.length-1){ cur++; paint(); }
  if(e.key.toLowerCase()==='m'){ $('mask').click(); }
  if(e.key.toLowerCase()==='b'){ $('box').click(); }
};

function csvCell(text){
  text=String(text==null?'':text);
  return /[",\n]/.test(text) ? '"'+text.replace(/"/g,'""')+'"' : text;
}
$('export').onclick=()=>{
  const rows=[['review_id','image','label_class','annotation_ok','notes'].join(',')];
  ITEMS.forEach(item=>{
    const r=state[item.id]||{};
    rows.push([item.id,item.image,r.label_class||'',r.annotation_ok||'',csvCell(r.notes||'')].join(','));
  });
  const blob=new Blob([rows.join('\n')+'\n'],{type:'text/csv'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob); a.download='review_sheet.csv'; a.click();
  URL.revokeObjectURL(a.href);
};
$('import').onchange=e=>{
  const file=e.target.files[0]; if(!file) return;
  const reader=new FileReader();
  reader.onload=()=>{
    const lines=reader.result.split(/\r?\n/).filter(Boolean).slice(1);
    lines.forEach(line=>{
      const cells=line.match(/("([^"]|"")*"|[^,]*)(,|$)/g).map(c=>
        c.replace(/,$/,'').replace(/^"|"$/g,'').replace(/""/g,'"'));
      const [id,,label,annotation,notes]=cells;
      if(id) state[id]={label_class:label||'',annotation_ok:annotation||'',notes:notes||''};
    });
    save(); paint();
  };
  reader.readAsText(file);
};

paint();
</script>
</body>
</html>
"""

if __name__=='__main__':
    build(sys.argv[1] if len(sys.argv)>1 else HERE.parent/'04_Hasil'/'benchmark_gen_v1')
