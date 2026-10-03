"""只读导入用户评分；保留工作簿、原文字及WPS内嵌图片，不修改Excel。"""
from pathlib import Path
import argparse, json, re, shutil, zipfile, posixpath
import xml.etree.ElementTree as ET
import openpyxl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--pipeline', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source, args.out/'打分记录.xlsx')
    shutil.copy2(args.pipeline, args.out/'gpt补景pipeline.md')
    book = openpyxl.load_workbook(args.source, read_only=True, data_only=False)
    raw = []
    for sheet in book:
        raw.append({'sheet': sheet.title, 'rows': [
            {'row': index, 'cells': [cell.value for cell in row]}
            for index, row in enumerate(sheet.iter_rows(), 1)
            if any(cell.value is not None for cell in row)]})
    ns = {'xdr': 'http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing',
          'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
          'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
    pictures = {}
    with zipfile.ZipFile(args.source) as archive:
        rels = {node.attrib['Id']: posixpath.normpath('xl/'+node.attrib['Target'])
                for node in ET.fromstring(archive.read('xl/_rels/cellimages.xml.rels'))}
        for pic in ET.fromstring(archive.read('xl/cellimages.xml')).findall('.//xdr:pic', ns):
            info = pic.find('.//xdr:cNvPr', ns)
            rid = pic.find('.//a:blip', ns).attrib['{'+ns['r']+'}embed']
            name = rels[rid].lstrip('/')
            filename = 'embedded/'+Path(name).name
            path = args.out/filename
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(archive.read(name))
            pictures[info.attrib['name']] = {'path': filename, 'description': info.attrib.get('descr')}
    table = next(sheet for sheet in raw if sheet['sheet'].startswith('r46 '))
    cases = []
    for row in table['rows'][1:]:
        values = row['cells']
        image_id = re.search(r'"(ID_[A-F0-9]+)"', values[6] or '')
        cases.append({'case_id': values[0].split(' · ')[0], 'label': values[0],
                      'human_score': values[1], 'RGB_prior_sufficient': values[2],
                      'OCC_prior_sufficient': values[3], 'kind': values[4], 'note': values[5],
                      'user_GPT_example': pictures.get(image_id.group(1)) if image_id else None,
                      'user_GPT_success_note': values[7],
                      'source': {'sheet': table['sheet'], 'row': row['row'], 'range': f'A{row["row"]}:H{row["row"]}'}})
    result = {'source_path': str(args.source), 'reviewer': 'human user',
              'supplied_conclusion': '当前真实DELETE：修复mask有收益；微调和新Adapter未建立明确收益。',
              'baseline': 'r46 official original weights + r21 full repaired SAM; adapter disabled',
              'cases': cases, 'all_sheets': raw, 'embedded_images': pictures,
              'blank_cells_remain_null': True, 'edited_source_workbook': False,
              'GPT_examples_scope': '用户提供的静态结果，不是本轮模型生成或视频通过率'}
    (args.out/'human_review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str)+'\n', encoding='utf-8')
    print(json.dumps({'sheets': len(raw), 'r46_cases': len(cases), 'real_scores': {
        c['case_id']: c['human_score'] for c in cases if c['kind']=='真实DELETE'}, 'embedded_images': len(pictures)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
