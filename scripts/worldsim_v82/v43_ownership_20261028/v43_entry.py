"""V43 query routing. An unedited request preserves the original V1 native path."""
from train_actor import render

def render_query(n,row,request,actor=None,frame=None,calibration=None):
    family=request.get('family','no_edit')
    if family=='no_edit':return n.render(row,dict(family='no_edit'))
    if family=='delete':return n.render(row,dict(family='delete'))
    assert actor is not None and frame is not None,'Edited vehicle needs its explicit local asset and observation time'
    assert family in ['translate','turn']
    if calibration is not None:
        from opacity_calibration import QueryActor
        actor=QueryActor(actor,calibration)
    return render(n,actor,frame,row['camera_index'],edit=dict(rotation=row['target_rotation_delta_world'],center=row['target_requested_center_world']))
