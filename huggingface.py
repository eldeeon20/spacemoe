"""HuggingFace integration: token handling, tokenizer upload/download, model checkpoint push."""

import os
import json
import time
import getpass
from pathlib import Path

from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError, EntryNotFoundError


class HFManager:
    def __init__(self, repo_id: str, revision: str = "main", token: str | None = None):
        self.repo_id = repo_id
        self.revision = revision
        self._token = token
        self._api = None

    def _get_token(self) -> str:
        if self._token:
            return self._token
        env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if env_token:
            return env_token
        print(f"\nNo HF token found. Enter token for {self.repo_id} (write access):")
        token = getpass.getpass("Token: ").strip()
        if not token:
            raise ValueError("Token required for HuggingFace operations")
        self._token = token
        return token

    def _get_api(self) -> HfApi:
        if self._api is None:
            self._api = HfApi(token=self._get_token())
        return self._api

    def ensure_repo(self):
        api = self._get_api()
        try:
            create_repo(repo_id=self.repo_id, exist_ok=True, private=False, token=self._get_token())
        except Exception:
            pass

    def ensure_revision(self):
        if self.revision in ("main", "master"):
            return
        api = self._get_api()
        try:
            api.create_branch(repo_id=self.repo_id, branch=self.revision, token=self._get_token())
            print(f"Created branch '{self.revision}' on {self.repo_id}")
        except Exception:
            pass

    def tokenizer_exists(self) -> bool:
        try:
            self._get_api().file_exists(
                repo_id=self.repo_id, filename="tokenizer.json", revision=self.revision
            )
            return True
        except (RepositoryNotFoundError, EntryNotFoundError):
            return False

    def download_tokenizer(self, local_path: str) -> str:
        """Download tokenizer.json from HF. Returns local path."""
        api = self._get_api()
        self.ensure_repo()
        try:
            path = hf_hub_download(
                repo_id=self.repo_id,
                filename="tokenizer.json",
                revision=self.revision,
                token=self._get_token(),
            )
            import shutil
            shutil.copy2(path, local_path)
            print(f"Downloaded tokenizer from {self.repo_id}@{self.revision} -> {local_path}")
            return local_path
        except Exception as e:
            print(f"Failed to download tokenizer from {self.repo_id}@{self.revision}: {e}")
            raise

    def upload_tokenizer(self, local_path: str, config_path: str | None = None):
        api = self._get_api()
        self.ensure_repo()
        self.ensure_revision()
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo="tokenizer.json",
            repo_id=self.repo_id,
            revision=self.revision,
            token=self._get_token(),
            commit_message="Add tokenizer",
        )
        if config_path and os.path.exists(config_path):
            api.upload_file(
                path_or_fileobj=config_path,
                path_in_repo="tokenizer_config.json",
                repo_id=self.repo_id,
                revision=self.revision,
                token=self._get_token(),
                commit_message="Add tokenizer config",
            )
        print(f"Uploaded tokenizer to {self.repo_id}@{self.revision}")

    def download_checkpoint(self, local_path: str, filename: str = "checkpoint.pt") -> bool:
        """Download checkpoint from HF. Returns True if successful."""
        try:
            api = self._get_api()
            path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                revision=self.revision,
                token=self._get_token(),
            )
            import shutil
            shutil.copy2(path, local_path)
            print(f"Downloaded {filename} from {self.repo_id}@{self.revision}")
            return True
        except Exception as e:
            print(f"Failed to download {filename} from {self.repo_id}@{self.revision}: {e}")
            return False

    def upload_checkpoint(self, checkpoint_path: str, safetensors_path: str | None = None,
                          tokenizer_path: str | None = None, step: int | None = None):
        api = self._get_api()
        self.ensure_repo()
        self.ensure_revision()
        files = [checkpoint_path]
        if safetensors_path and os.path.exists(safetensors_path):
            files.append(safetensors_path)
        if tokenizer_path and os.path.exists(tokenizer_path):
            files.append(tokenizer_path)
            if os.path.exists("tokenizer_config.json"):
                files.append("tokenizer_config.json")

        msg = f"Checkpoint step {step}" if step else f"Checkpoint {time.strftime('%Y-%m-%d %H:%M')}"
        api.upload_file(
            path_or_fileobj=files[0],
            path_in_repo=os.path.basename(files[0]),
            repo_id=self.repo_id,
            revision=self.revision,
            token=self._get_token(),
            commit_message=msg,
        )
        for f in files[1:]:
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=os.path.basename(f),
                repo_id=self.repo_id,
                revision=self.revision,
                token=self._get_token(),
                commit_message=msg,
            )
        print(f"Uploaded {len(files)} files to {self.repo_id}@{self.revision} (step {step})")


class PeriodicPusher:
    """Upload checkpoint every N minutes."""
    def __init__(self, hf_manager: HFManager, interval_minutes: int = 10):
        self.hf = hf_manager
        self.interval = interval_minutes * 60
        self.last_push = time.time()

    def maybe_push(self, checkpoint_path: str, safetensors_path: str | None,
                   tokenizer_path: str | None, step: int) -> bool:
        if time.time() - self.last_push >= self.interval:
            self.hf.upload_checkpoint(checkpoint_path, safetensors_path, tokenizer_path, step)
            self.last_push = time.time()
            return True
        return False