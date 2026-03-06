from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def lazy_import_requests():
	import requests  # type: ignore

	return requests


def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> None:
	dest_path.parent.mkdir(parents=True, exist_ok=True)
	requests = lazy_import_requests()

	with requests.get(url, stream=True, timeout=60) as resp:
		resp.raise_for_status()
		tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
		with open(tmp_path, "wb") as f:
			for chunk in resp.iter_content(chunk_size=chunk_size):
				if chunk:
					f.write(chunk)
		os.replace(tmp_path, dest_path)


def download_first_available(urls: Iterable[str], dest_path: Path) -> str:
	last_error: Exception | None = None
	for url in urls:
		try:
			download_file(url, dest_path)
			return url
		except Exception as e:
			last_error = e
	if last_error is None:
		raise RuntimeError("No URLs provided")
	raise last_error
