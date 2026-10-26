"""世界坐标证据与编辑版本；生成假设不会自动升级为真实观测。"""
import json
import sqlite3
from pathlib import Path


class WorldState:
    def __init__(self, path, scene):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS scene(id TEXT PRIMARY KEY, master TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS revision(
          scene TEXT, id TEXT, parent TEXT, payload TEXT NOT NULL,
          PRIMARY KEY(scene,id));
        CREATE TABLE IF NOT EXISTS surface(
          scene TEXT, id INTEGER, xyz TEXT, status TEXT, source_frames TEXT,
          source_count INTEGER, repair_revision TEXT, PRIMARY KEY(scene,id));
        CREATE TABLE IF NOT EXISTS request(
          scene TEXT, id TEXT, revision TEXT, operation TEXT, output TEXT,
          PRIMARY KEY(scene,id));
        CREATE TABLE IF NOT EXISTS hypothesis(
          scene TEXT, id TEXT, revision TEXT, backend TEXT, payload TEXT,
          PRIMARY KEY(scene,id));
        ''')
        self.scene = scene

    def bind(self, master):
        value = json.dumps(master, sort_keys=True)
        row = self.db.execute('SELECT master FROM scene WHERE id=?', (self.scene,)).fetchone()
        if row and row[0] != value:
            raise ValueError('主副本变化，必须登记新场景版本')
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO scene VALUES(?,?)', (self.scene, value))

    def revision(self, key, parent, payload):
        value = json.dumps(payload, sort_keys=True)
        row = self.db.execute('SELECT parent,payload FROM revision WHERE scene=? AND id=?', (self.scene,key)).fetchone()
        if row and row != (parent,value):
            raise ValueError('同一个版本ID不能覆盖为不同编辑状态')
        if parent and not self.db.execute('SELECT 1 FROM revision WHERE scene=? AND id=?',(self.scene,parent)).fetchone():
            raise ValueError('缺少父版本')
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO revision VALUES(?,?,?,?)',(self.scene,key,parent,value))

    def surfaces(self, rows):
        with self.db:
            self.db.executemany('INSERT OR IGNORE INTO surface VALUES(?,?,?,?,?,?,?)',
                [(self.scene,int(i),json.dumps(xyz),status,json.dumps(src),int(count),revision)
                 for i,xyz,status,src,count,revision in rows])

    def record_request(self, key, revision, operation, output):
        row = (self.scene,key,revision,json.dumps(operation,sort_keys=True),str(output))
        previous = self.db.execute('SELECT * FROM request WHERE scene=? AND id=?',(self.scene,key)).fetchone()
        if previous and previous != row:
            raise ValueError('请求ID冲突')
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO request VALUES(?,?,?,?,?)',row)

    def summary(self):
        return dict(scene=self.scene,
            surfaces_by_status=dict(self.db.execute('SELECT status,count(*) FROM surface WHERE scene=? GROUP BY status',(self.scene,))),
            revisions=self.db.execute('SELECT count(*) FROM revision WHERE scene=?',(self.scene,)).fetchone()[0],
            requests=self.db.execute('SELECT count(*) FROM request WHERE scene=?',(self.scene,)).fetchone()[0],
            generated_hypotheses=self.db.execute('SELECT count(*) FROM hypothesis WHERE scene=?',(self.scene,)).fetchone()[0],
            generated_is_observed=False, geometric_validity_of_repair='unverified')

    def close(self):
        self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        self.db.close()
