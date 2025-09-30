import os
import socket
from pathlib import Path
import sys

BLOCKLIST_EXT = {'.pt', '.pth', '.ckpt'}


def internet_available(host: str = '8.8.8.8', port: int = 53, timeout: float = 1.0) -> bool:
	 try:
		 socket.setdefaulttimeout(timeout)
		 socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
		 return True
	 except Exception:
		 return False


def external_checkpoints_exist(root: Path) -> bool:
	 for p in root.rglob('*'):
		 if p.suffix.lower() in BLOCKLIST_EXT and 'outputs' not in str(p).lower():
			 return True
	 return False


def main():
	 if internet_available():
		 print('Offline guard: internet detected. Abort.')
		 sys.exit(2)
	 repo = Path(__file__).resolve().parents[1]
	 if external_checkpoints_exist(repo):
		 print('Offline guard: external checkpoints detected in repo. Abort.')
		 sys.exit(3)
	 print('Offline guard: OK')


if __name__ == '__main__':
	 main()
