"""PyTorch 2.4 的 eager 编译后端不接受 mode；只去掉该无效优化选项。"""
def apply():
    from nerfstudio.utils import misc
    original=misc.torch_compile
    def compile_compatible(*args,**kwargs):
        if kwargs.get('backend')=='eager':kwargs.pop('mode',None)
        return original(*args,**kwargs)
    misc.torch_compile=compile_compatible
