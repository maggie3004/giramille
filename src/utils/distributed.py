import os
import torch
import torch.distributed as dist


def is_distributed() -> bool:
	 return dist.is_available() and dist.is_initialized()


def setup_ddp(backend: str = "nccl") -> None:
	 if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
		 rank = int(os.environ["RANK"])
		 world_size = int(os.environ["WORLD_SIZE"])
		 dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
		 torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


def cleanup_ddp() -> None:
	 if is_distributed():
		 dist.barrier()
		 dist.destroy_process_group()


def get_rank() -> int:
	 return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
	 return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
	 return get_rank() == 0
