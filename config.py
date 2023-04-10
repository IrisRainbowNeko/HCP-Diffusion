from colossalai.amp import AMP_TYPE

gradient_clipping = 1.0
#gradient_accumulation = 1

fp16 = dict(
    mode=AMP_TYPE.TORCH,
)

#parallel = dict(
#    pipeline=dict(size=2),
#    tensor=dict(size=4, mode='2d')
#)