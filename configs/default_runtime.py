checkpoint_config = dict(interval=1)

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),

    ])

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
