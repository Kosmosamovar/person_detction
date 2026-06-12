# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['web_cam_person_detection.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('config.json', '.'),
        ('yolov8n.pt', '.'),
        ('yolov5n.pt', '.'),
        ('yolov5s.pt', '.'),
        ('yolov8n.onnx', '.'),
        ('yolov5n.onnx', '.'),
        ('MobileNetSSD_deploy.caffemodel', '.'),
    ],
    hiddenimports=['torch.distributed', 'torch.distributed.algorithms'],
    hookspath=['.'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch.distributed.optim',
        'torch.distributed.algorithms._optimizer_overlap',
        'torch.distributed.elastic',
        'torch.distributed.launch',
        'torch.distributed.run',
        'torch.utils.tensorboard',
        'tensorboard',
        'sqlalchemy',
        'flask_sqlalchemy',
        'alembic',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='web_cam_person_detection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
