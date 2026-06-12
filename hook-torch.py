from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

# Custom safe hook: keep torch.distributed modules, skip only known crashing optimizer overlap submodule.
module_collection_mode = 'pyz+py'

datas = collect_data_files(
    'torch',
    excludes=[
        '**/*.h',
        '**/*.hpp',
        '**/*.cuh',
        '**/*.lib',
        '**/*.cpp',
        '**/*.pyi',
        '**/*.cmake',
    ],
)

hiddenimports = collect_submodules(
    'torch',
    filter=lambda name: not (
        name.startswith('torch.distributed.optim')
        or name.startswith('torch.distributed.algorithms._optimizer_overlap')
    )
)
if 'torch.distributed' not in hiddenimports:
    hiddenimports.append('torch.distributed')
if 'torch.distributed.algorithms' not in hiddenimports:
    hiddenimports.append('torch.distributed.algorithms')

binaries = collect_dynamic_libs('torch')
