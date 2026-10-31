import hashlib,json,posixpath,zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

BASE=Path(__file__).resolve().parent
XLSX=Path('C:/Users/dengyunlong/Downloads/editor_badcase_analysis.xlsx')
DOC=Path('C:/Users/dengyunlong/Downloads/V43_METHOD_PROPOSAL_R4.md')
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
with zipfile.ZipFile(XLSX) as z:
    strings=[]
    if 'xl/sharedStrings.xml' in z.namelist():
        strings=[''.join(n.itertext()) for n in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',NS)]
    rels={n.attrib['Id']:n.attrib['Target'] for n in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
    sheets=[]
    for sheet in ET.fromstring(z.read('xl/workbook.xml')).findall('s:sheets/s:sheet',NS):
        target=rels[sheet.attrib['{'+NS['r']+'}id']]
        path=posixpath.normpath('xl/'+target) if not target.startswith('/') else target.lstrip('/')
        root=ET.fromstring(z.read(path)); cells=[]
        for c in root.findall('.//s:sheetData/s:row/s:c',NS):
            value=c.find('s:v',NS);typ=c.attrib.get('t');text=value.text if value is not None else None
            if typ=='s':text=strings[int(text)]
            elif typ=='inlineStr':text=''.join(c.find('s:is',NS).itertext())
            formula=c.find('s:f',NS)
            if text is not None or formula is not None:cells.append(dict(cell=c.attrib['r'],value=text,formula=None if formula is None else formula.text))
        sheets.append(dict(name=sheet.attrib['name'],xml=path,cells=cells,hyperlinks=[n.attrib for n in root.findall('.//s:hyperlink',NS)]))
    media=[dict(path=n,bytes=z.getinfo(n).file_size) for n in z.namelist() if n.startswith('xl/media/')]
    relationships=[]
    for n in z.namelist():
        if n.endswith('.rels') and n!='xl/_rels/workbook.xml.rels':
            relationships.append(dict(path=n,entries=[v.attrib for v in ET.fromstring(z.read(n))]))
    data=dict(schema='read_only_xlsx_source_extraction_v1',source=str(XLSX),sha256=hashlib.sha256(XLSX.read_bytes()).hexdigest(),sheets=sheets,media=media,relationships=relationships)
    (BASE/'badcase_workbook_extract.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for sheet in sheets:
        print('SHEET',sheet['name'])
        for cell in sheet['cells']:print(json.dumps(cell,ensure_ascii=True))
    print('MEDIA',json.dumps(media));print('RELATIONSHIPS',json.dumps(relationships,ensure_ascii=True))
lines=DOC.read_text(encoding='utf-8').splitlines();start=next(i for i,s in enumerate(lines) if s.startswith('## 3.'));end=next(i for i in range(start+1,len(lines)) if lines[i].startswith('## 4.'))
(BASE/'r4_section3_source.txt').write_text('\n'.join(f'{i+1}: {lines[i]}' for i in range(start,end))+'\n',encoding='utf-8')
print('R4_SECTION3_LINES',start+1,end)
