from __future__ import annotations
import json, os
from typing import Any, Dict
from datetime import datetime

class JSONLRunLogger:
    def __init__(self, path: str = "runs.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        record = {"ts": datetime.utcnow().isoformat(), **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")