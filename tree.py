#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目录树生成器 - 类似于tree命令的Python实现
功能：展示当前目录的完整文件结构
作者：Assistant
"""

import os
import argparse
from pathlib import Path
from typing import List, Set, Optional
import sys


class DirectoryTree:
    def __init__(self,
                 show_hidden: bool = False,
                 max_depth: Optional[int] = None,
                 ignore_patterns: Optional[List[str]] = None,
                 only_dirs: bool = False,
                 show_size: bool = False):
        """
        初始化目录树生成器

        Args:
            show_hidden: 是否显示隐藏文件/文件夹
            max_depth: 最大显示深度
            ignore_patterns: 要忽略的文件/文件夹模式列表
            only_dirs: 是否只显示目录
            show_size: 是否显示文件大小
        """
        self.show_hidden = show_hidden
        self.max_depth = max_depth
        self.ignore_patterns = ignore_patterns or []
        self.only_dirs = only_dirs
        self.show_size = show_size

        # 统计信息
        self.dir_count = 0
        self.file_count = 0
        self.total_size = 0

        # 树状图符号
        self.tree_symbols = {
            'branch': '├── ',
            'last': '└── ',
            'vertical': '│   ',
            'space': '    '
        }

    def should_ignore(self, path: Path) -> bool:
        """判断是否应该忽略某个路径"""
        name = path.name

        # 隐藏文件处理
        if not self.show_hidden and name.startswith('.'):
            return True

        # 常见的需要忽略的文件夹/文件
        default_ignore = {
            '__pycache__', '.git', '.svn', '.hg',
            'node_modules', '.vscode', '.idea',
            '*.pyc', '*.pyo', '*.pyd', '.DS_Store',
            'Thumbs.db', '*.log'
        }

        # 检查是否匹配忽略模式
        for pattern in (self.ignore_patterns + list(default_ignore)):
            if pattern.startswith('*') and name.endswith(pattern[1:]):
                return True
            elif name == pattern:
                return True

        return False

    def get_file_size(self, path: Path) -> str:
        """获取文件大小的人类可读格式"""
        try:
            size = path.stat().st_size
            self.total_size += size

            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024:
                    return f" ({size:.1f}{unit})"
                size /= 1024
            return f" ({size:.1f}TB)"
        except:
            return ""

    def generate_tree(self, directory: Path, prefix: str = "", depth: int = 0) -> List[str]:
        """
        递归生成目录树

        Args:
            directory: 要遍历的目录
            prefix: 当前行的前缀
            depth: 当前深度

        Returns:
            包含目录树每一行的字符串列表
        """
        if self.max_depth is not None and depth > self.max_depth:
            return []

        try:
            # 获取目录下的所有项目
            items = []
            for item in directory.iterdir():
                if not self.should_ignore(item):
                    if self.only_dirs and not item.is_dir():
                        continue
                    items.append(item)

            # 按名称排序，目录在前
            items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))

        except PermissionError:
            return [f"{prefix}[权限拒绝]"]

        lines = []

        for index, item in enumerate(items):
            is_last = index == len(items) - 1

            # 选择合适的符号
            if is_last:
                current_prefix = prefix + self.tree_symbols['last']
                next_prefix = prefix + self.tree_symbols['space']
            else:
                current_prefix = prefix + self.tree_symbols['branch']
                next_prefix = prefix + self.tree_symbols['vertical']

            # 构建显示名称
            display_name = item.name

            if item.is_dir():
                display_name = f"📁 {display_name}/"
                self.dir_count += 1

                # 添加目录项
                lines.append(f"{current_prefix}{display_name}")

                # 递归处理子目录
                if depth < (self.max_depth or float('inf')):
                    sub_lines = self.generate_tree(item, next_prefix, depth + 1)
                    lines.extend(sub_lines)
            else:
                # 根据文件扩展名添加图标
                icon = self.get_file_icon(item.suffix.lower())
                size_info = self.get_file_size(item) if self.show_size else ""
                display_name = f"{icon} {display_name}{size_info}"
                self.file_count += 1

                lines.append(f"{current_prefix}{display_name}")

        return lines

    def get_file_icon(self, extension: str) -> str:
        """根据文件扩展名返回对应的图标"""
        icon_map = {
            '.py': '🐍',
            '.js': '🟨',
            '.html': '🌐',
            '.css': '🎨',
            '.json': '📋',
            '.xml': '📄',
            '.yml': '⚙️',
            '.yaml': '⚙️',
            '.md': '📝',
            '.txt': '📄',
            '.pdf': '📕',
            '.doc': '📘',
            '.docx': '📘',
            '.xls': '📗',
            '.xlsx': '📗',
            '.ppt': '📙',
            '.pptx': '📙',
            '.zip': '🗜️',
            '.rar': '🗜️',
            '.tar': '🗜️',
            '.gz': '🗜️',
            '.jpg': '🖼️',
            '.jpeg': '🖼️',
            '.png': '🖼️',
            '.gif': '🖼️',
            '.svg': '🖼️',
            '.mp4': '🎬',
            '.avi': '🎬',
            '.mov': '🎬',
            '.mp3': '🎵',
            '.wav': '🎵',
            '.cpp': '⚡',
            '.c': '⚡',
            '.java': '☕',
            '.go': '🐹',
            '.rs': '🦀',
            '.php': '🐘',
            '.rb': '💎',
            '.sh': '🖥️',
            '.bat': '🖥️',
            '.exe': '⚙️',
        }
        return icon_map.get(extension, '📄')

    def format_size(self, size_bytes: int) -> str:
        """格式化字节大小为人类可读格式"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    def print_tree(self, directory: str = "."):
        """打印目录树"""
        root_path = Path(directory).resolve()

        print(f"📂 {root_path.name}/ ({root_path})")

        # 生成树状结构
        tree_lines = self.generate_tree(root_path)

        # 打印树状结构
        for line in tree_lines:
            print(line)

        # 打印统计信息
        print(f"\n📊 统计信息:")
        print(f"   📁 目录: {self.dir_count} 个")
        if not self.only_dirs:
            print(f"   📄 文件: {self.file_count} 个")
            if self.show_size:
                print(f"   💾 总大小: {self.format_size(self.total_size)}")


def main():
    parser = argparse.ArgumentParser(
        description="显示目录树结构 - 类似于tree命令",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python directory_tree.py                    # 显示当前目录树
  python directory_tree.py /path/to/dir       # 显示指定目录树
  python directory_tree.py -a                 # 显示包括隐藏文件
  python directory_tree.py -d 2               # 只显示2层深度
  python directory_tree.py -D                 # 只显示目录
  python directory_tree.py -s                 # 显示文件大小
  python directory_tree.py -i "*.pyc,node_modules"  # 忽略特定文件/文件夹
        """
    )

    parser.add_argument('directory', nargs='?', default='.',
                        help='要显示的目录路径 (默认: 当前目录)')
    parser.add_argument('-a', '--all', action='store_true',
                        help='显示隐藏文件和目录')
    parser.add_argument('-d', '--depth', type=int, metavar='N',
                        help='限制显示深度为N层')
    parser.add_argument('-D', '--dirs-only', action='store_true',
                        help='只显示目录')
    parser.add_argument('-s', '--size', action='store_true',
                        help='显示文件大小')
    parser.add_argument('-i', '--ignore', type=str, metavar='PATTERNS',
                        help='忽略的文件/目录模式，用逗号分隔')

    args = parser.parse_args()

    # 处理忽略模式
    ignore_patterns = []
    if args.ignore:
        ignore_patterns = [pattern.strip() for pattern in args.ignore.split(',')]

    # 创建目录树生成器
    tree = DirectoryTree(
        show_hidden=args.all,
        max_depth=args.depth,
        ignore_patterns=ignore_patterns,
        only_dirs=args.dirs_only,
        show_size=args.size
    )

    # 检查目录是否存在
    if not os.path.exists(args.directory):
        print(f"❌ 错误: 目录 '{args.directory}' 不存在")
        sys.exit(1)

    if not os.path.isdir(args.directory):
        print(f"❌ 错误: '{args.directory}' 不是一个目录")
        sys.exit(1)

    # 生成并打印目录树
    try:
        tree.print_tree(args.directory)
    except KeyboardInterrupt:
        print("\n\n⏹️  已取消")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()