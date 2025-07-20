#!/usr/bin/env python3
"""
脚本功能：
1. 检查是否安装 falkordb-bulk-loader，否则通过 pip 安装
2. 从 RocksDB（路径 /tmp/rag/）导出图数据为 CSV（nodes.csv, edges.csv）
3. 调用 falkordb-bulk-loader 将 CSV 批量导入 FalkorDB
"""
import os
import sys
import subprocess
import csv
import rocksdb
from graph_pb2 import NodeAttributes, EdgeAttributes

# RocksDB 路径
DB_PATH = "/tmp/rag/"
# 导出文件路径
NODES_CSV = "nodes.csv"
EDGES_CSV = "edges.csv"

# ColumnFamily 名称
CF_DEFAULT = b'default'
CF_NODE = b'node_attr'
CF_EDGE = b'edge_attr'

# 检查并安装 falkordb-bulk-loader
def ensure_bulk_loader():
    try:
        subprocess.run(["falkordb-bulk-loader", "--help"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("falkordb-bulk-loader 未检测到，正在安装...")
        subprocess.run([sys.executable, "-m", "pip", "install", "falkordb-bulk-loader"], check=True)
        print("安装完成。")

# 导出节点和边到 CSV
def export_to_csv():
    # 打开 RocksDB 并获取 CF handles
    opts = rocksdb.Options(create_if_missing=False)
    opts.create_if_missing_column_families = False
    db = rocksdb.DB(DB_PATH, opts)
    cf_handles = {name: db.get_column_family(name) for name in [CF_DEFAULT, CF_NODE, CF_EDGE]}

    # 导出节点属性
    with open(NODES_CSV, 'w', newline='', encoding='utf-8') as f_nodes:
        writer = csv.writer(f_nodes)
        # CSV header，根据 Protobuf 定义字段
        writer.writerow(['node_id', 'label', 'age', 'active'])
        it = db.iterkeys(cf=cf_handles[CF_NODE])
        it.seek_to_first()
        for key in it:
            raw = db.get(key, cf=cf_handles[CF_NODE])
            msg = NodeAttributes()
            msg.ParseFromString(raw)
            writer.writerow([
                key.decode(),
                msg.label,
                msg.age,
                msg.active,
            ])

    # 导出边及邻接列表同时导出边属性
    with open(EDGES_CSV, 'w', newline='', encoding='utf-8') as f_edges:
        writer = csv.writer(f_edges)
        # CSV header
        writer.writerow(['src', 'tgt', 'weight', 'since'])
        # 遍历邻接列表
        it = db.iteritems(cf=cf_handles[CF_DEFAULT])
        it.seek_to_first()
        for key, raw_list in it:
            src = key.decode()
            neighbors = raw_list.decode().split(',') if raw_list else []
            for tgt in neighbors:
                # 尝试读取边属性
                edge_key = key + b'->' + tgt.encode()
                raw_edge = db.get(edge_key, cf=cf_handles[CF_EDGE])
                if raw_edge:
                    msg_e = EdgeAttributes()
                    msg_e.ParseFromString(raw_edge)
                    writer.writerow([src, tgt, msg_e.weight, msg_e.since])
                else:
                    # 若无属性，写空置
                    writer.writerow([src, tgt, '', ''])

# 使用 falkordb-bulk-loader 导入
def bulk_load():
    cmd = [
        "falkordb-bulk-loader",
        "--nodes", NODES_CSV,
        "--edges", EDGES_CSV
    ]
    print("执行导入命令：", ' '.join(cmd))
    subprocess.run(cmd, check=True)
    print("批量导入完成。")


def main():
    ensure_bulk_loader()
    print("开始导出 CSV...")
    export_to_csv()
    print(f"导出完成: {NODES_CSV}, {EDGES_CSV}")
    print("开始批量导入到 FalkorDB...")
    bulk_load()

if __name__ == '__main__':
    main()
