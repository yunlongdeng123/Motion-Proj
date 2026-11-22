"""Replace this research line's historical status tail, preserving other tasks byte-exact."""
MARKER='## ROAD-GAUSSIAN-TRANSFER-20261122'
LABEL='ROAD-VEHICLE-JOINT-20261123'

def replace_status_tail(old,replacement):
    if not isinstance(replacement,str) or LABEL not in replacement:
        raise ValueError('Root must provide status_replacement for the current task')
    lines=replacement.strip().splitlines()
    if sum(line.startswith('## ') for line in lines)!=1 or not lines[0].startswith('## '):
        raise ValueError('RESEARCH_STATUS needs one current section with history navigation, not appended historical sections')
    offsets=[i for i in range(len(old)) if old.startswith(MARKER,i) and (i==0 or old[i-1]=='\n') and (i+len(MARKER)==len(old) or old[i+len(MARKER)] in '\r\n')]
    if len(offsets)!=1:raise ValueError('Expected one exact original lineage-tail marker')
    start=offsets[0];prefix=old[:start];result=prefix+replacement.strip()+'\n'
    if result[:start]!=prefix:raise AssertionError('Unrelated current-status prefix changed')
    return result,dict(replacement_from_marker=MARKER,unrelated_prefix_characters=len(prefix),
        replaced_tail_characters=len(old)-start,current_sections=1,historical_origin_body_retained_in_version_history=True)
