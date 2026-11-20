import csv
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from PIL import Image
from melon_cyclegan import texture_study as s

class ReviewTests(unittest.TestCase):
    def fixture(self,t):
        study=Path(t)/'study';scope=study/'p100';dataset=Path(t)/'data';dataset.mkdir()
        rows=[]
        for i,cls in enumerate([0,0,0,1,1]):
            pid=f'parent{i}';Image.new('RGB',(32,32),(10+i,100,30)).save(dataset/f'{pid}.jpg')
            (dataset/f'{pid}.txt').write_text(f'{cls} .3 .4 .2 .3\n')
            rows.append(dict(id=pid,class_id=str(cls),split='train',image=f'{pid}.jpg',label=f'{pid}.txt',photograph_group=pid))
        valid=dict(rows[0],id='val',split='valid')
        m=dict(rows=rows,validation=[valid],subset_digest='subset',crops_digest='crop')
        s.write(scope/'subset.json',m)
        for arm in s.GEN_ARMS:
            b=scope/'candidates'/arm;(b/'images').mkdir(parents=True);(b/'full').mkdir()
            rec=[]
            for i in range(3):
                Image.new('RGB',(32,32),(10,100,30)).save(b/f'images/{i}.png')
                Image.new('RGB',(32,32),(10,100,30)).save(b/f'full/{i}.png')
                (b/f'full/{i}.txt').write_text('1 .3 .4 .2 .3\n')
                rec.append(dict(candidate_id=str(i),parent_id=f'parent{i}',image=f'images/{i}.png',full_image=f'full/{i}.png',full_label=f'full/{i}.txt'))
            s.write(b/'records.json',rec);s.write(b/'binding.json',dict(subset_digest='subset',generator_sha256='weight'))
            s.write(scope/'diagnostics'/f'{arm}.json',dict(subset_digest='subset',generator_sha256='weight',synthetic=dict(chroma_energy=.1)))
        return study,scope,dataset,m

    def complete(self,scope,reject_arm=None):
        key={r['blind_id']:r for r in s.read(scope/'review_key_PRIVATE.json')}
        file=scope/'review/review.csv'
        with file.open() as f:rows=list(csv.DictReader(f));fields=list(rows[0])
        for r in rows:
            r.update(target_label='unripe',texture_realistic='yes',shape_ok='yes',boundary_ok='yes',artifact_free='yes',accepted='yes',reviewer='test reviewer')
            if key[r['blind_id']]['arm']==reject_arm:r['accepted']='no'
        with file.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

    def test_review_binding_selection_and_counts(self):
        with tempfile.TemporaryDirectory() as t:
            study,scope,dataset,m=self.fixture(t)
            with patch.object(s,'verify_scope',return_value=(scope,m)):
                s.blind_review(study,100,s.GEN_ARMS)
                self.assertNotIn('G0',(scope/'review/index.html').read_text())
                with self.assertRaisesRegex(ValueError,'incomplete'):s.reviewed(study,100)
                self.complete(scope)
                self.assertEqual(s.select(study)['winner'],'G1')
                counts=s.build_detectors(study,100,dataset)
                self.assertEqual(counts['real_only']['added'],0)
                self.assertTrue(all(counts[a]['added']==1 for a in s.DET_ARMS if a!='real_only'))
                self.assertEqual((scope/'detector_arms/geometry/labels/train/add_0000.txt').read_text().split()[1],'0.7')
                (scope/'candidates/G0/full/0.txt').write_text('0 .3 .4 .2 .3\n')
                with self.assertRaisesRegex(ValueError,'changed'):s.reviewed(study,100)

    def test_rejected_candidates_stop_selection(self):
        with tempfile.TemporaryDirectory() as t:
            study,scope,dataset,m=self.fixture(t)
            with patch.object(s,'verify_scope',return_value=(scope,m)):
                s.blind_review(study,100,s.GEN_ARMS);self.complete(scope)
                file=scope/'review/review.csv'
                file.write_text(file.read_text().replace(',yes,test reviewer',',no,test reviewer'))
                with self.assertRaisesRegex(ValueError,'no viable'):s.select(study)

if __name__=='__main__':unittest.main()
