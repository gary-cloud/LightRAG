import os
from typing import Any, final
from dataclasses import dataclass
import pipmaster as pm
import configparser
from contextlib import asynccontextmanager

if not pm.is_installed("falkordb"):
    pm.install("falkordb")

# aioredis is a depricated library, replaced with redis
from redis.asyncio import ConnectionPool  # type: ignore
from redis.exceptions import RedisError, ConnectionError  # type: ignore

from falkordb.asyncio import *  # type: ignore
from falkordb.exceptions import *  # type: ignore
from falkordb.edge import Edge
from falkordb.node import Node

from lightrag.utils import logger

from lightrag.base import BaseGraphStorage
from lightrag.types import KnowledgeGraph

import json

# Get maximum number of graph nodes from environment variable, default is 1000
MAX_GRAPH_NODES = int(os.getenv("MAX_GRAPH_NODES", 1000))

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")

# Constants for Redis connection pool
MAX_CONNECTIONS = 50
SOCKET_TIMEOUT = 5.0
SOCKET_CONNECT_TIMEOUT = 3.0


@final
@dataclass
class RedisGraphStorage(BaseGraphStorage):
    def __post_init__(self):
        redis_url = os.environ.get(
            "REDIS_URI", config.get("redis", "uri", fallback="redis://localhost:6379")
        )
        # Create a connection pool with limits
        self._pool = ConnectionPool.from_url(
            redis_url,
            max_connections=MAX_CONNECTIONS,
            decode_responses=True,
            socket_timeout=SOCKET_TIMEOUT,
            socket_connect_timeout=SOCKET_CONNECT_TIMEOUT,
        )
        self._falkordb = FalkorDB(connection_pool=self._pool)
        logger.info(
            f"Initialized Redis connection pool for {self.namespace} with max {MAX_CONNECTIONS} connections"
        )

    @asynccontextmanager
    async def _get_falkordb_connection(self):
        """Safe context manager for Redis operations."""
        try:
            yield self._falkordb
        except ConnectionError as e:
            logger.error(f"Redis connection error in {self.namespace}: {e}")
            raise
        except RedisError as e:
            logger.error(f"Redis operation error in {self.namespace}: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error in Redis operation for {self.namespace}: {e}"
            )
            raise

    async def __aenter__(self):
        """Support for async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure Redis resources are cleaned up when exiting context."""
        await self._pool.aclose()
    
    async def has_node(self, node_id: str) -> bool:
        """
        Check if a node with the given label exists in the database

        Args:
            node_id: Label of the node to check

        Returns:
            bool: True if node exists, False otherwise

        Raises:
            ValueError: If node_id is invalid
            Exception: If there is an error executing the query
        """
        async with self._get_falkordb_connection() as falkordb:
            g = falkordb.select_graph('RAG')
            if not node_id:
                raise ValueError("Node ID cannot be empty")
            try:
                query = f"MATCH (n) WHERE n.id = $node_id RETURN COUNT(n) > 0 AS nodeExists"
                result = await g.query(query, params={"node_id": node_id})
                return result[0][0]
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """
        Check if an edge exists between two nodes

        Args:
            source_node_id: Label of the source node
            target_node_id: Label of the target node

        Returns:
            bool: True if edge exists, False otherwise

        Raises:
            ValueError: If either node_id is invalid
            Exception: If there is an error executing the query
        """
        async with self._get_falkordb_connection() as falkordb:
            g = falkordb.select_graph('RAG')
            if not source_node_id or not target_node_id:
                raise ValueError("Source and target node IDs cannot be empty")
            try:
                query = f"""
                MATCH (source)-[e]->(target)
                WHERE source.id = $source AND target.id = $target
                RETURN COUNT(e) > 0 AS edgeExists
                """
                result = await g.query(query, params={"source": source_node_id, "target": target_node_id})
                return result[0][0]
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its label identifier, return only node properties

        Args:
            node_id: The node label to look up

        Returns:
            dict: Node properties if found
            None: If node not found

        Raises:
            ValueError: If node_id is invalid
            Exception: If there is an error executing the query
        """
        async with self._get_falkordb_connection() as falkordb:
            g = falkordb.select_graph('RAG')
            if not node_id:
                raise ValueError("Node ID cannot be empty")
            try:
                query = f"MATCH (n) WHERE n.id = $node_id RETURN n"
                result = await g.query(query, params={"node_id": node_id})
                if result:
                    return result.result_set[0][0].properties
                return None
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """
        Retrieve multiple nodes in one query using UNWIND.

        Args:
            node_ids: List of node entity IDs to fetch.

        Returns:
            A dictionary mapping each node_id to its node data (or None if not found).
        """
        async with self._get_falkordb_connection() as falkordb:
            g = falkordb.select_graph('RAG')
            if not node_ids:
                return {}
            try:
                query = f"""
                UNWIND $node_ids AS id
                MATCH (n) WHERE n.id = id
                RETURN n
                """
                result = await g.query(query, params={"node_ids": node_ids})
                ret = []
                for row in result:
                    if row[0] is not None:
                        ret.append(row[0].properties)
                return ret
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise
    
    async def node_degree(self, node_id: str) -> int:
        """Get the degree (number of relationships) of a node with the given label.
        If multiple nodes have the same label, returns the degree of the first node.
        If no node is found, returns 0.

        Args:
            node_id: The label of the node

        Returns:
            int: The number of relationships the node has, or 0 if no node found

        Raises:
            ValueError: If node_id is invalid
            Exception: If there is an error executing the query
        """
        async with self._get_falkordb_connection() as falkordb:
            g = falkordb.select_graph('RAG')
            if not node_id:
                raise ValueError("Node ID cannot be empty")
            try:
                query = f"""
                MATCH (n) WHERE n.id = $node_id
                RETURN DEGREE(n) AS degree
                """
                result = await g.query(query, params={"node_id": node_id})
                return result[0][0] if result else 0
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """
        Retrieve the degree for multiple nodes in a single query using UNWIND.

        Args:
            node_ids: List of node labels (entity_id values) to look up.

        Returns:
            A dictionary mapping each node_id to its degree (number of relationships).
            If a node is not found, its degree will be set to 0.
        """
        async with self._get_falkordb_connection() as falkordb:
            g = falkordb.select_graph('RAG')
            if not node_ids:
                return {}
            try:
                query = f"""
                UNWIND $node_ids AS id
                MATCH (n) WHERE n.id = id
                RETURN DEGREE(n) AS degree
                """
                result = await g.query(query, params={"node_ids": node_ids})
                ret = []
                for row in result:
                    if row[0] is not None:
                        ret.append(row[0])
                return ret
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree (sum of relationships) of two nodes.

        Args:
            src_id: Label of the source node
            tgt_id: Label of the target node

        Returns:
            int: Sum of the degrees of both nodes
        """
        src_degree = await self.node_degree(src_id)
        trg_degree = await self.node_degree(tgt_id)

        # Convert None to 0 for addition
        src_degree = 0 if src_degree is None else src_degree
        trg_degree = 0 if trg_degree is None else trg_degree

        degrees = int(src_degree) + int(trg_degree)
        return degrees
    
    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        """
        Calculate the combined degree for each edge (sum of the source and target node degrees)
        in batch using the already implemented node_degrees_batch.

        Args:
            edge_pairs: List of (src, tgt) tuples.

        Returns:
            A dictionary mapping each (src, tgt) tuple to the sum of their degrees.
        """
        # Collect unique node IDs from all edge pairs.
        unique_node_ids = {src for src, _ in edge_pairs}
        unique_node_ids.update({tgt for _, tgt in edge_pairs})

        # Get degrees for all nodes in one go.
        degrees = await self.node_degrees_batch(list(unique_node_ids))

        # Sum up degrees for each edge pair.
        edge_degrees = {}
        for src, tgt in edge_pairs:
            edge_degrees[(src, tgt)] = degrees.get(src, 0) + degrees.get(tgt, 0)
        return edge_degrees
    
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        """Get edge properties between two nodes.

        Args:
            source_node_id: Label of the source node
            target_node_id: Label of the target node

        Returns:
            dict: Edge properties if found, default properties if not found or on error

        Raises:
            ValueError: If either node_id is invalid
            Exception: If there is an error executing the query
        """
        async with self._get_falkordb_connection() as falkordb:
            g = falkordb.select_graph('RAG')
            if not source_node_id or not target_node_id:
                raise ValueError("Source and target node IDs cannot be empty")
            try:
                query = f"""
                MATCH (source)-[e]->(target)
                WHERE source.id = $source AND target.id = $target
                RETURN e
                """
                result = await g.query(query, params={"source": source_node_id, "target": target_node_id})
                if result:
                    return result.result_set[0][0].properties
                return None
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        """
        Retrieve edge properties for multiple (src, tgt) pairs in one query.

        Args:
            pairs: List of dictionaries, e.g. [{"src": "node1", "tgt": "node2"}, ...]

        Returns:
            A dictionary mapping (src, tgt) tuples to their edge properties.
        """
        async with self._get_falkordb_connection() as falkordb:
            g = falkordb.select_graph('RAG')
            if not pairs:
                return {}
            try:
                query = "UNWIND $pairs AS pair " \
                        "MATCH (source)-[e]->(target) " \
                        "WHERE source.id = pair.src AND target.id = pair.tgt " \
                        "RETURN e"
                result = await g.query(query, {"pairs": pairs})
                ret = []
                for row in result:
                    if row[0] is not None:
                        ret.append(row[0].properties)
                return ret
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Retrieves all edges (relationships) for a particular node identified by its label.

        Args:
            source_node_id: Label of the node to get edges for

        Returns:
            list[tuple[str, str]]: List of (source_label, target_label) tuples representing edges
            None: If no edges found

        Raises:
            ValueError: If source_node_id is invalid
            Exception: If there is an error executing the query
        """
        async with self._get_falkordb_connection() as falkordb:
            g = falkordb.select_graph('RAG')
            if not source_node_id:
                raise ValueError("Source node ID cannot be empty")
            try:
                query = f"""
                MATCH (source)-[e]->(target)
                WHERE source.id = $source
                RETURN source.id AS src, target.id AS tgt
                """
                result = await g.query(query, params={"source": source_node_id})
                ret = []
                for row in result:
                    if row[0] is not None and row[1] is not None:
                        ret.append((row[0], row[1]))
                return ret
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """
        Batch retrieve edges for multiple nodes in one query using UNWIND.
        For each node, returns both outgoing and incoming edges to properly represent
        the undirected graph nature.

        Args:
            node_ids: List of node IDs (entity_id) for which to retrieve edges.

        Returns:
            A dictionary mapping each node ID to its list of edge tuples (source, target).
            For each node, the list includes both:
            - Outgoing edges: (queried_node, connected_node)
            - Incoming edges: (connected_node, queried_node)
        """
        async with self._get_falkordb_connection() as falkordb:
            g = falkordb.select_graph('RAG')
            if not node_ids:
                return {}
            try:
                query = f"""
                UNWIND $node_ids AS id
                MATCH (n)-[e]->(m)
                WHERE n.id = id OR m.id = id
                RETURN n.id AS src, m.id AS tgt
                """
                result = await g.query(query, params={"node_ids": node_ids})
                edges_dict = {}
                for row in result:
                    src, tgt = row[0], row[1]
                    if src not in edges_dict:
                        edges_dict[src] = []
                    if tgt not in edges_dict:
                        edges_dict[tgt] = []
                    edges_dict[src].append((src, tgt))
                    edges_dict[tgt].append((src, tgt))
                return edges_dict
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """
        Upsert a node in the Neo4j database.

        Args:
            node_id: The unique identifier for the node (used as label)
            node_data: Dictionary of node properties
        """
        properties = node_data
        entity_type = properties["entity_type"]
        if "entity_id" not in properties:
            raise ValueError("Node data must contain 'entity_id' property")
        
        try:
            async with self._get_falkordb_connection() as falkordb:
                g = falkordb.select_graph('RAG')
                query = f"""
                MERGE (n:{entity_type} {{id: $node_id}})
                SET n += $properties
                """
                await g.query(query, {"properties": properties, "node_id": node_id})
                logger.debug(f"Upserted node {node_id} with properties {properties}")
        except Exception as e:
            logger.error(f"Error during upsert: {str(e)}")
            raise

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """
        Upsert an edge and its properties between two nodes identified by their labels.
        Ensures both source and target nodes exist and are unique before creating the edge.
        Uses entity_id property to uniquely identify nodes.

        Args:
            source_node_id (str): Label of the source node (used as identifier)
            target_node_id (str): Label of the target node (used as identifier)
            edge_data (dict): Dictionary of properties to set on the edge

        Raises:
            ValueError: If either source or target node does not exist or is not unique
        """
        async with self._get_falkordb_connection() as falkordb:
            g = falkordb.select_graph('RAG')
            if not source_node_id or not target_node_id:
                raise ValueError("Source and target node IDs cannot be empty")
            try:
                query = f"""
                MATCH (source {{id: $source}}), (target {{id: $target}})
                MERGE (source)-[r:DIRECTED]->(target)
                SET r += $edge_data
                """
                await g.query(query, {"edge_data": edge_data, "source": source_node_id, "target": target_node_id})
                logger.debug(
                    f"Upserted edge from {source_node_id} to {target_node_id} with properties {edge_data}"
                )
            except Exception as e:
                logger.error(f"Error during upsert: {str(e)}")
                raise

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = MAX_GRAPH_NODES,
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the label includes the specified `node_label`.

        Args:
            node_label: Label of the starting node, * means all nodes
            max_depth: Maximum depth of the subgraph, Defaults to 3
            max_nodes: Maxiumu nodes to return by BFS, Defaults to 1000

        Returns:
            KnowledgeGraph object containing nodes and edges, with an is_truncated flag
            indicating whether the graph was truncated due to max_nodes limit
        """
        async with self._get_falkordb_connection() as falkordb:
            g = falkordb.select_graph('RAG')
            try:
                query = f"""
                MATCH (n:$node_label)-[:DIRECTED*1..$max_depth]->(m)
                WITH n, m LIMIT $max_nodes
                RETURN n, m
                """
                result = await g.query(query, params={"node_label": node_label, "max_depth": max_depth, "max_nodes": max_nodes})
                
                nodes = {}
                edges = []
                for row in result:
                    src_node = row[0]
                    tgt_node = row[1]
                    if src_node.id not in nodes:
                        nodes[src_node.id] = src_node.properties
                    if tgt_node.id not in nodes:
                        nodes[tgt_node.id] = tgt_node.properties
                    edges.append((src_node.id, tgt_node.id))
                
                is_truncated = len(nodes) >= max_nodes
                return KnowledgeGraph(nodes=nodes, edges=edges, is_truncated=is_truncated)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

    async def get_all_labels(self) -> list[str]:
        """
        Get all existing node labels in the database
        Returns:
            ["Person", "Company", ...]  # Alphabetically sorted label list
        """
        async with self._get_falkordb_connection() as falkordb:
            g = falkordb.select_graph('RAG')
            try:
                query = "CALL db.labels()"
                result = await g.query(query)
                labels = []
                for row in result:
                    if row[0] is not None:
                        labels.append(row[0])
                return labels
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

    async def delete_node(self, node_id: str) -> None:
        """Delete a node with the specified label

        Args:
            node_id: The label of the node to delete
        """
        async with self._get_falkordb_connection() as falkordb:
            g = falkordb.select_graph('RAG')
            if not node_id:
                raise ValueError("Node ID cannot be empty")
            try:
                query = f"MATCH (n) WHERE n.id = $node_id DETACH DELETE n"
                await g.query(query, params={"node_id": node_id})
                logger.info(f"Deleted node {node_id}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes

        Args:
            nodes: List of node labels to be deleted
        """
        async with self._get_falkordb_connection() as falkordb:
            g = falkordb.select_graph('RAG')
            if not nodes:
                return
            try:
                query = f"""
                UNWIND {nodes} AS id
                MATCH (n) WHERE n.id = id
                DETACH DELETE n
                """
                await g.query(query)
                logger.info(f"Deleted {len(nodes)} nodes from {self.namespace}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """
        async with self._get_falkordb_connection() as falkordb:
            g = falkordb.select_graph('RAG')
            if not edges:
                return
            try:
                for src, tgt in edges:
                    query = f"""
                    MATCH (source)-[e]->(target)
                    WHERE source.id = $src AND target.id = $tgt
                    DELETE e
                    """
                    await g.query(query, params={"src": src, "tgt": tgt})
                logger.info(f"Deleted {len(edges)} edges from {self.namespace}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

    async def drop(self) -> dict[str, str]:
        """Drop all data from storage and clean up resources

        This method will delete all nodes and relationships in the Neo4j database.

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}
        """
        async with self._get_falkordb_connection() as falkordb:
            g = falkordb.select_graph('RAG')
            try:
                query = "MATCH (n) DETACH DELETE n"
                await g.query(query)
                logger.info(f"Dropped all data from {self.namespace}")
                return {"status": "success", "message": "data dropped"}
            except Exception as e:
                logger.error(f"Error dropping data from {self.namespace}: {e}")
                return {"status": "error", "message": str(e)}