"""Verify the delivered files, relative links, media streams, and input hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import cv2
import numpy as np
from PIL import Image, ImageSequence

HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths=[]
        self.video_count=0
        self.ids=set()

    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        for key in ('href','src','poster'):
            if key in attrs: self.paths.append(attrs[key])
        if tag=='video': self.video_count+=1
        if 'id' in attrs: self.ids.add(attrs['id'])


def main(*, allow_missing_inputs: bool = False):
    report={'material_date':'2026-10-07','checks':{}}
    checks=report['checks']
    manifest=json.loads((HERE/'素材元数据.json').read_text(encoding='utf-8'))
    images=sorted((HERE/'images').glob('*.png'))
    videos=sorted((HERE/'videos').glob('*.mp4'))
    gifs=sorted((HERE/'videos').glob('*.gif'))
    posters=sorted((HERE/'videos').glob('*_poster.jpg'))
    assert (len(images),len(videos),len(gifs),len(posters))==(10,4,4,4)
    checks['counts']={'images':10,'mp4':4,'gif':4,'poster':4}
    checks['image_formats']=[]
    for path in images:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            assert im.size==(1600,1000) and im.mode=='RGB'
            checks['image_formats'].append({'file':path.name,'size':list(im.size),'format':im.format})
    for path in posters:
        with Image.open(path) as im:
            assert im.size==(1280,720)
            im.verify()
    checks['videos']=[]
    for path in videos:
        cap=cv2.VideoCapture(str(path));assert cap.isOpened()
        fps=cap.get(cv2.CAP_PROP_FPS)
        declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc=int(cap.get(cv2.CAP_PROP_FOURCC))
        codec=bytes((fourcc>>(8*i))&255 for i in range(4)).decode('ascii',errors='replace')
        first,last=None,None
        count=0; sample_checks=[]
        while True:
            ok,frame=cap.read()
            if not ok: break
            assert frame.shape==(720,1280,3)
            if count in (0,75,149):
                roi=frame[245:529,65:1215]
                assert float(roi.std())>10, f'{path.name} blank body frame'
                sample_checks.append(count)
            if first is None: first=frame.copy()
            last=frame
            count+=1
        cap.release()
        assert count==declared==150 and abs(fps-15)<.01
        assert sample_checks==[0,75,149]
        # Check the media contains movement in the visual area, not just a changing HUD.
        motion=float(np.mean(np.abs(first[245:529,65:1215].astype(float)-last[245:529,65:1215])))
        assert motion>.5, f'{path.name} does not show pose changes'
        assert codec in ('h264','H264','avc1','mp4v','FMP4'), codec
        checks['videos'].append({'file':path.name,'decoded_frames':count,'fps':fps,'duration_seconds':count/fps,
                                 'codec':codec,'sample_frames_checked':sample_checks,'body_mean_abs_change':motion})
    checks['gifs']=[]
    for path in gifs:
        with Image.open(path) as im:
            assert im.size==(640,360) and im.is_animated and im.info.get('loop')==0
            duration=sum(frame.info.get('duration',0) for frame in ImageSequence.Iterator(im))
            assert abs(duration-10000)<=200
            checks['gifs'].append({'file':path.name,'frames':im.n_frames,'duration_ms':duration,'loop':0})
    checks['output_hashes_match_manifest']=all(sha(HERE/item['relative_path'])==item['sha256'] for item in manifest['outputs'])
    assert checks['output_hashes_match_manifest']
    checks['protected_inputs']=[]
    missing_inputs=[]
    for item in manifest['inputs']:
        path=Path(item['original_path'])
        if not path.is_file():
            if not allow_missing_inputs:
                raise FileNotFoundError(
                    f'Protected input is unavailable: {path}. '
                    'Restore it or pass --allow-missing-inputs to verify the remaining deliverables.'
                )
            missing_inputs.append(str(path))
            checks['protected_inputs'].append(
                {'name': path.name, 'sha256': item['sha256'], 'available': False, 'unchanged': None}
            )
            continue
        assert sha(path)==item['sha256'], f'Input modified: {path}'
        checks['protected_inputs'].append(
            {'name': path.name, 'sha256': item['sha256'], 'available': True, 'unchanged': True}
        )
    parser=References();parser.feed((HERE/'index.html').read_text(encoding='utf-8'))
    assert parser.video_count==4
    references=[('index.html',value) for value in parser.paths]
    for doc in HERE.glob('*.md'):
        content=doc.read_text(encoding='utf-8')
        for match in re.finditer(r'!?\[[^\]]*\]\(([^)]+)\)',content):
            references.append((doc.name,match.group(1)))
    checked=0
    for origin,target in references:
        parsed=urlsplit(target)
        assert not parsed.scheme and not parsed.netloc, f'External dependency in {origin}: {target}'
        if not parsed.path:
            assert origin=='index.html' and parsed.fragment in parser.ids
            continue
        dest=(HERE/unquote(parsed.path)).resolve()
        assert HERE in dest.parents, f'Link escapes portable present folder: {target}'
        if dest.name!='验收结果.json':
            assert dest.is_file(), f'Broken link in {origin}: {target}'
        checked+=1
    checks['relative_links']={'checked':checked,'html_video_elements':parser.video_count,'broken':0,'external_dependencies':0}
    checks['evidence']={'tests':manifest['evidence']['tests'],
                       'valid_frames':{side:manifest['evidence']['cpu_gpu_comparison'][side]['valid'] for side in ['left','right']}}
    # Human inspection is described separately; never claim it from a parser check.
    report['visual_review']='Recorded separately in 素材说明.md; this script checks structural and numeric validity only.'
    checks['protected_inputs_missing']=missing_inputs
    report['passed']=True
    (HERE/'验收结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'passed':True,'counts':checks['counts'],'relative_links':checks['relative_links'],
                      'video_codecs':[item['codec'] for item in checks['videos']],
                      'protected_inputs_unchanged':not missing_inputs,
                      'protected_inputs_missing':missing_inputs},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Verify portable deliverable structure and media.')
    parser.add_argument(
        '--allow-missing-inputs',
        action='store_true',
        help='verify deliverables without the original D:\\ source inputs',
    )
    main(**vars(parser.parse_args()))
