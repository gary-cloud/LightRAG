import os
from typing import Any, final
from dataclasses import dataclass
import pipmaster as pm
import asyncio
import rocksdb
from typing import Any, Dict, List, Tuple, Optional
from contextlib import asynccontextmanager

if not pm.is_installed("python-rocksdb"):
    pm.install("python-rocksdb")

from lightrag.utils import logger

from lightrag.base import BaseGraphStorage
from lightrag.types import KnowledgeGraph

# 三个 ColumnFamily 名称
CF_NAMES = [b'default', b'node_attr', b'edge_attr']

class GraphMergeOperator(rocksdb.interfaces.MergeOperator):
    def name(self) -> bytes:
        return b"LSMGraphMergeOperator"

    def full_merge(self, key: bytes, existing: Optional[bytes], ops: List[bytes]):
        edges = set()
        if existing:
            edges.update(int(x) for x in existing.decode().split(',') if x)
        for op in ops:
            text = op.decode()
            if not text:
                continue
            flag, rest = text[0], text[1:]
            ids = {int(x) for x in rest.split(',') if x}
            if flag == '+':
                edges |= ids
            elif flag == '-':
                edges -= ids
            else:
                edges |= ids
        merged = sorted(edges)
        return True, ",".join(str(x) for x in merged).encode()

    def partial_merge(self, key: bytes, left: bytes, right: bytes):
        return self.full_merge(key, None, [left, right])

@final
@dataclass
class LsmGraphStorage(BaseGraphStorage):
    path: str = "/tmp/rag"
    namespace: str = "default"

    def __post_init__(self):
        opts = rocksdb.Options()
        opts.create_if_missing = True
        opts.create_if_missing_column_families = True
        opts.merge_operator = GraphMergeOperator()
        cfs = {name: rocksdb.ColumnFamilyDescriptor(name, opts) for name in CF_NAMES}
        self._db = rocksdb.DB(self.path, opts, column_families=list(cfs.values()))
        # mapping names to handles
        self._cf_handles = dict(zip(CF_NAMES, self._db.column_families))

    @asynccontextmanager
    async def _get_db(self):
        """Async context manager (no-op, for interface parity)"""
        yield self._db

    async def has_node(self, node_id: str) -> bool:
        if not node_id:
            raise ValueError("Node ID cannot be empty")
        key = node_id.encode()
        data = await asyncio.to_thread(self._db.get, key, cf=self._cf_handles[b'node_attr'])
        return data is not None

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        if not source_node_id or not target_node_id:
            raise ValueError("Node IDs cannot be empty")
        key = source_node_id.encode() + b'->' + target_node_id.encode()
        data = await asyncio.to_thread(self._db.get, key, cf=self._cf_handles[b'edge_attr'])
        return data is not None

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        if not node_id:
            raise ValueError("Node ID cannot be empty")
        key = node_id.encode()
        raw = await asyncio.to_thread(self._db.get, key, cf=self._cf_handles[b'node_attr'])
        if raw is None:
            return None
        msg = NodeAttributes()
        msg.ParseFromString(raw)
        return { 'label': msg.label, 'age': msg.age, 'active': msg.active }

    async def get_nodes_batch(self, node_ids: List[str]) -> Dict[str, Dict]:
        result: Dict[str, Dict] = {}
        tasks = []
        for nid in node_ids:
            tasks.append(self.get_node(nid))
        responses = await asyncio.gather(*tasks)
        for nid, props in zip(node_ids, responses):
            result[nid] = props
        return result

    async def node_degree(self, node_id: str) -> int:
        if not node_id:
            raise ValueError("Node ID cannot be empty")
        # adjacency list stored under default CF
        raw = await asyncio.to_thread(self._db.get, node_id.encode(), cf=self._cf_handles[b'default'])
        if not raw:
            return 0
        edges = raw.decode().split(',')
        return len(edges)

    async def node_degrees_batch(self, node_ids: List[str]) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for nid in node_ids:
            deg = await self.node_degree(nid)
            result[nid] = deg
        return result

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        src_deg = await self.node_degree(src_id)
        tgt_deg = await self.node_degree(tgt_id)
        return src_deg + tgt_deg

    async def edge_degrees_batch(self, edge_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], int]:
        # collect unique nodes
        unique = set()
        for s, t in edge_pairs:
            unique.update([s, t])
        node_degs = await self.node_degrees_batch(list(unique))
        return { (s,t): node_degs.get(s,0) + node_degs.get(t,0) for s,t in edge_pairs }

    async def get_edge(self, source_node_id: str, target_node_id: str) -> Optional[Dict[str, Any]]:
        if not source_node_id or not target_node_id:
            raise ValueError("Node IDs cannot be empty")
        key = source_node_id.encode() + b'->' + target_node_id.encode()
        raw = await asyncio.to_thread(self._db.get, key, cf=self._cf_handles[b'edge_attr'])
        if raw is None:
            return None
        msg = EdgeAttributes()
        msg.ParseFromString(raw)
        return { 'weight': msg.weight, 'since': msg.since }

    async def get_edges_batch(self, pairs: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict]:
        result: Dict[Tuple[str,str], Dict] = {}
        for entry in pairs:
            src, tgt = entry['src'], entry['tgt']
            props = await self.get_edge(src, tgt)
            result[(src,tgt)] = props
        return result

    async def get_node_edges(self, source_node_id: str) -> List[Tuple[str, str]]:
        if not source_node_id:
            raise ValueError("Node ID cannot be empty")
        raw = await asyncio.to_thread(self._db.get, source_node_id.encode(), cf=self._cf_handles[b'default'])
        if not raw:
            return []
        neighbors = [nid for nid in raw.decode().split(',') if nid]
        return [(source_node_id, nid) for nid in neighbors]

    async def get_nodes_edges_batch(self, node_ids: List[str]) -> Dict[str, List[Tuple[str, str]]]:
        result: Dict[str, List[Tuple[str,str]]] = {}
        for nid in node_ids:
            edges = await self.get_node_edges(nid)
            result[nid] = edges
        return result

    async def upsert_node(self, node_id: str, node_data: Dict[str, Any]) -> None:
        if 'entity_id' not in node_data:
            raise ValueError("Node data must contain 'entity_id'")
        msg = NodeAttributes(
            label=node_data.get('label', ''),
            age=int(node_data.get('age', 0)),
            active=bool(node_data.get('active', False))
        )
        await asyncio.to_thread(self._db.put, node_id.encode(), msg.SerializeToString(), cf=self._cf_handles[b'node_attr'])

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: Dict[str, Any]) -> None:
        key = source_node_id.encode() + b'->' + target_node_id.encode()
        msg = EdgeAttributes(
            weight=float(edge_data.get('weight', 0.0)),
            since=edge_data.get('since', '')
        )
        await asyncio.to_thread(self._db.put, key, msg.SerializeToString(), cf=self._cf_handles[b'edge_attr'])
        # 同时更新邻接列表
        await asyncio.to_thread(self._db.merge, source_node_id.encode(), b'+' + target_node_id.encode(), cf=self._cf_handles[b'default'])

    async def get_knowledge_graph(self, node_label: str, max_depth: int = 3, max_nodes: int = 1000) -> KnowledgeGraph:
        raise NotImplementedError("Not implemented")
        # BFS
        visited = set()
        queue = [(node_label, 0)]
        nodes = {}
        edges = []
        while queue and len(nodes) < max_nodes:
            current, depth = queue.pop(0)
            if current in visited or depth > max_depth:
                continue
            visited.add(current)
            props = await self.get_node(current)
            if props is not None:
                nodes[current] = props
            nbrs = await self.get_node_edges(current)
            for _, tgt in nbrs:
                edges.append((current, tgt))
                queue.append((tgt, depth+1))
        return KnowledgeGraph(nodes=nodes, edges=edges, is_truncated=len(nodes) >= max_nodes)

    async def get_all_labels(self) -> List[str]:
        # RocksDB 无模式，需扫描 node_attr CF
        it = self._db.iterate(cf=self._cf_handles[b'node_attr'])
        it.seek_to_first()
        labels = set()
        for k, v in it:
            msg = NodeAttributes()
            msg.ParseFromString(v)
            labels.add(msg.label)
        return sorted(labels)

    async def delete_node(self, node_id: str) -> None:
        await asyncio.to_thread(self._db.delete, node_id.encode(), cf=self._cf_handles[b'node_attr'])
        # 同时删除邻接列表
        await asyncio.to_thread(self._db.delete, node_id.encode(), cf=self._cf_handles[b'default'])

    async def remove_nodes(self, nodes: List[str]) -> None:
        for nid in nodes:
            await self.delete_node(nid)

    async def remove_edges(self, edges: List[Tuple[str, str]]) -> None:
        for src, tgt in edges:
            key = src.encode() + b'->' + tgt.encode()
            await asyncio.to_thread(self._db.delete, key, cf=self._cf_handles[b'edge_attr'])
            # 从邻接列表移除
            await asyncio.to_thread(self._db.merge, src.encode(), b'-' + tgt.encode(), cf=self._cf_handles[b'default'])

    async def drop(self) -> Dict[str, str]:
        # 删除所有 CF 数据
        for cf in CF_NAMES:
            it = self._db.iterate(cf=self._cf_handles[cf])
            it.seek_to_first()
            batch = rocksdb.WriteBatch()
            for k, _ in it:
                batch.delete(cf=cf, key=k)
            await asyncio.to_thread(self._db.write, batch)
        return {"status": "success", "message": "data dropped"}
