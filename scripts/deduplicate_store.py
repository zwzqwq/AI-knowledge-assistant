"""
向量库去重维护脚本

背景：向量库曾出现 96% 重复切片（同一文档被多次导入，每次 add() 都追加副本）。
本脚本原地去重：每个唯一文本只保留第一个副本，其余全部删除。

用法：
  python scripts/deduplicate_store.py            # 预览（dry-run，不删）
  python scripts/deduplicate_store.py --execute  # 执行删除

安全设计：
  1. 默认 dry-run，只统计打印，不碰数据
  2. 只删"内容完全重复"的切片（和保留的副本一字不差）→ 不丢任何内容
  3. 删除后自动验证：总数 = 唯一数，且唯一文本集合不变
"""
import argparse
import os
import sys
from collections import Counter

# 脚本在 scripts/ 下运行，把项目根目录加进搜索路径才能 import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import logger
from src.rag.retriever import Retriever


def main():
    parser = argparse.ArgumentParser(description="向量库去重维护脚本")
    parser.add_argument("--execute", action="store_true", help="真正执行删除（默认只预览）")
    args = parser.parse_args()

    r = Retriever()
    vs = r._get_or_load()
    if vs is None:
        logger.error("向量库未初始化，无需清理")
        sys.exit(1)

    collection = vs._collection
    data = collection.get()
    ids, docs = data["ids"], data["documents"]

    # 按文本统计出现次数
    counter = Counter(docs)
    unique_count = len(counter)
    print(f"总切片: {len(docs)} | 唯一文本: {unique_count} | "
          f"重复文本: {sum(1 for n in counter.values() if n > 1)} 个")

    # 找出要删的 id：每个唯一文本保留第一个，其余副本全删
    seen = set()
    to_delete = []
    for cid, text in zip(ids, docs):
        if text in seen:
            to_delete.append(cid)
        else:
            seen.add(text)

    print(f"将保留: {len(seen)} 个切片 | 将删除: {len(to_delete)} 个重复副本")
    if not to_delete:
        print("✅ 没有重复，无需清理")
        return

    if not args.execute:
        print("\n[dry-run] 未做任何修改。确认无误后加 --execute 执行。")
        return

    # 执行删除（Chroma 支持按 id 批量删）
    collection.delete(ids=to_delete)
    print(f"已删除 {len(to_delete)} 个重复切片")

    # 验证：总数收敛到唯一数，且唯一文本集合不变
    after = collection.get()
    after_counter = Counter(after["documents"])
    ok_count = len(after["documents"]) == unique_count
    ok_text = set(after["documents"]) == set(counter.keys())
    print(f"删除后总切片: {len(after['documents'])}（期望 {unique_count}）")
    print(f"唯一文本集合未变: {'✅' if ok_text else '❌'}")
    if ok_count and ok_text:
        print("✅ 清理完成，向量库已去重")
    else:
        print("⚠️ 验证未通过，请检查！")


if __name__ == "__main__":
    main()
