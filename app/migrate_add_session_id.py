"""数据库迁移脚本 - 为todo_items表添加session_id字段"""

import sqlite3
from pathlib import Path

# 数据库文件路径
DB_PATH = Path(__file__).parent / "data" / "lifeops.db"

def add_session_id_to_todos():
    """为todo_items表添加session_id字段"""
    
    if not DB_PATH.exists():
        print(f"❌ 数据库文件不存在: {DB_PATH}")
        return False
    
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        # 检查session_id字段是否已存在
        cursor.execute("PRAGMA table_info(todo_items)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if "session_id" in columns:
            print("✅ session_id 字段已存在，无需迁移")
            conn.close()
            return True
        
        # 添加session_id字段
        print("🔄 正在添加 session_id 字段...")
        cursor.execute("""
            ALTER TABLE todo_items 
            ADD COLUMN session_id VARCHAR(64)
        """)
        
        # 为session_id创建索引（提高查询性能）
        print("🔄 正在创建 session_id 索引...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_todo_items_session_id 
            ON todo_items(session_id)
        """)
        
        conn.commit()
        conn.close()
        
        print("✅ 数据库迁移成功！")
        print("   - 已添加 session_id 字段到 todo_items 表")
        print("   - 已创建 session_id 索引")
        return True
        
    except Exception as e:
        print(f"❌ 迁移失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("LifeOps-Agent 数据库迁移工具")
    print("=" * 50)
    success = add_session_id_to_todos()
    if success:
        print("\n✨ 迁移完成！请重启应用。")
    else:
        print("\n⚠️  迁移失败，请检查错误信息。")
