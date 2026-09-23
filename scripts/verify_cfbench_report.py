"""验证离线报告链接、原始视频来源与真实生成数量，不计算质量总分。"""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import av


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids = [], set()
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'id' in a:
            self.ids.add(a['id'])
        for key in ['src', 'href']:
            if key in a:
                self.links.append(a[key])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    data = json.loads((args.report/'report-data.json').read_text())
    parser = Links()
    parser.feed((args.report/'index.html').read_text())
    for link in parser.links:
        if link.startswith(('https://', 'http://')):
            continue
        if link.startswith('#'):
            assert link[1:] in parser.ids, link
        else:
            assert (args.report/link).is_file(), link
    assert data['generated']==data['automatic_readout']==data['ai_reviewed']==len(data['items'])==24
    videos = []
    for item in data['items']:
        cid, row = item['case_id'], item['input']
        assert item['human_verdict'] is None
        assert item['ai_preliminary']['human_verdict'] is None
        root = args.report/'cases'/cid
        provenance = json.loads((root/'original-nuscenes.json').read_text())
        case = row['case']
        first = case['anchor']['event_frame']-case['anchor']['pre_frames']
        assert provenance['source_frames']==list(range(first,first+24))
        assert provenance['camera_index']==row['camera_index']
        assert provenance['scene_id']==case['dataset']['scene_id']
        for branch, expected, size, fps in [('original-nuscenes',24,(1600,900),10),('factual',77,(1280,704),30),('counterfactual',77,(1280,704),30)]:
            with av.open(str(root/f'{branch}.mp4')) as reader:
                stream = reader.streams.video[0]
                assert float(stream.average_rate)==fps
                assert (stream.width,stream.height)==size
                count = sum(1 for _ in reader.decode(video=0))
                assert count==expected,(cid,branch,count)
            videos.append({'case_id':cid,'branch':branch,'decoded_frames':count})
    result = {'status':'pass','checked_cases':24,'decoded_videos':len(videos),'internal_links_checked':sum(not x.startswith(('https://','http://')) for x in parser.links),
              'human_verdicts_filled':0,'videos':videos}
    (args.report/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='videos'}))


if __name__=='__main__':
    main()
