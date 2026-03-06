from __future__ import annotations

import json
import mysql.connector
from pathlib import Path
from typing import List, Tuple, Dict, Union

import numpy as np

from .embedder_mobilefacenet_arcface import l2_normalize


class FaceDatabase:
    """
    Database Interface using MySQL.
    Stores face embeddings and labels in a relational table.
    """
    def __init__(self, config: Dict):
        """
        :param config: Dictionary containing mysql connection info 
                       (host, user, password, database)
        """
        self.config = config
        self.conn = None
        self.cursor = None
        self._init_db()
        self._load_cache()

    def _init_db(self):
        """Initialize MySQL database connection and table."""
        try:
            # 1. Connect to MySQL Server (assume DB might not exist yet)
            # We connect to 'information_schema' or no DB to create the target DB first
            temp_config = self.config.copy()
            target_db = temp_config.pop('database', 'face_recognition')
            
            # Connect without DB to ensure it exists
            conn = mysql.connector.connect(**temp_config)
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {target_db}")
            conn.close()
            
            # 2. Connect to the specific DB
            self.config['database'] = target_db
            self.conn = mysql.connector.connect(**self.config)
            self.cursor = self.conn.cursor()
            
            # 3. Create table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS faces (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    label VARCHAR(255) NOT NULL,
                    embedding LONGBLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.commit()
            print(f"Connected to MySQL database: {target_db}")
            
        except mysql.connector.Error as err:
            raise RuntimeError(f"MySQL Connection Error: {err}")

    def _load_cache(self):
        """Load all embeddings into memory for fast matching."""
        self.cursor.execute("SELECT label, embedding FROM faces")
        rows = self.cursor.fetchall()
        
        self.labels = []
        embeddings_list = []
        
        for label, emb_bytes in rows:
            self.labels.append(label)
            # MySQL BLOB -> numpy array
            emb_array = np.frombuffer(emb_bytes, dtype=np.float32)
            embeddings_list.append(emb_array)
            
        if embeddings_list:
            self.embeddings = np.vstack(embeddings_list)
        else:
            self.embeddings = np.zeros((0, 0), dtype=np.float32)

    def close(self):
        if self.conn:
            self.conn.close()

    def add(self, label: str, embedding: np.ndarray) -> None:
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        
        # 1. Save to DB (Persistent Storage)
        emb_bytes = embedding.tobytes()
        
        sql = "INSERT INTO faces (label, embedding) VALUES (%s, %s)"
        val = (label, emb_bytes)
        
        self.cursor.execute(sql, val)
        self.conn.commit()
        
        # 2. Update Memory Cache (Runtime Speed)
        self._load_cache()

    def is_empty(self) -> bool:
        return self.embeddings.shape[0] == 0

    def match(self, embedding: np.ndarray, threshold: float = 0.35) -> Tuple[str, float]:
        """Return (label, score). If score < threshold => unknown."""
        if self.is_empty():
            return "unknown", 0.0

        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        embedding = l2_normalize(embedding)

        if embedding.shape[0] != self.embeddings.shape[1]:
            print(f"Warning: Dim mismatch {embedding.shape[0]} vs {self.embeddings.shape[1]}")
            return "unknown", 0.0

        scores = self.embeddings @ embedding
        
        # Group scores by label (Max pooling)
        label_scores = {}
        for idx, score in enumerate(scores):
            lbl = self.labels[idx]
            if lbl not in label_scores or score > label_scores[lbl]:
                label_scores[lbl] = score
        
        if not label_scores:
            return "unknown", 0.0
            
        best_label = max(label_scores, key=label_scores.get)
        best_score = float(label_scores[best_label])

        if best_score < threshold:
            return "unknown", best_score
        return best_label, best_score
