"""mpi4py wrapper module (simplified and always enables mpi4py if available)."""

import os

verbose = False
has_key = lambda key: key in os.environ.keys()

# 是否显式禁用 PlanckLens MPI
has_key = lambda key: key in os.environ.keys()

# 是否显式禁用 PlanckLens MPI
if has_key('USE_PLANCKLENS_MPI'):
    val = os.environ['USE_PLANCKLENS_MPI'].lower()
    if val in ['1', 'true', 'yes']:
        use = True
    elif val in ['0', 'false', 'no']:
        use = False
    else:
        raise ValueError(f"Invalid USE_PLANCKLENS_MPI value: {os.environ['USE_PLANCKLENS_MPI']}")
else:
    use = True

# 尝试启用 mpi4py
if use:
    try:
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        barrier = MPI.COMM_WORLD.Barrier
        bcast = MPI.COMM_WORLD.bcast
        send = MPI.COMM_WORLD.send
        receive = MPI.COMM_WORLD.recv
        finalize = MPI.Finalize
        ANY_SOURCE = MPI.ANY_SOURCE

        if verbose:
            print(f'mpi.py: setup OK, rank {rank} in {size}')
    except ImportError:
        # 无法导入 mpi4py，退化为单进程模式
        rank = 0
        size = 1
        barrier = lambda: None
        bcast = lambda _: _
        send = lambda _, dest: None
        receive = lambda _, source: None
        finalize = lambda: None
        ANY_SOURCE = 0
        if verbose:
            print('mpi.py: unable to import mpi4py, running in serial mode')
else:
    # 显式禁用 MPI
    rank = 0
    size = 1
    barrier = lambda: None
    bcast = lambda _: _
    send = lambda _, dest: None
    receive = lambda _, source: None
    finalize = lambda: None
    ANY_SOURCE = 0
    if verbose:
        print('mpi.py: Plancklens.mpi disabled by environment variable')
