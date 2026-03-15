#!/usr/bin/env python3
"""
Recursively delete .docx, .txt, and .json files from a folder and its subfolders.

Usage:
    python cleanup_outputs.py <folder_path> [--dry-run] [--yes]

Options:
    --dry-run    Show what would be deleted without actually deleting
    --yes        Skip confirmation prompt and delete immediately
"""

import sys
import os
from pathlib import Path
from typing import List


def find_files_to_delete(root_folder: Path) -> List[Path]:
    """Find all .docx, .txt, and .json files recursively."""
    extensions = {'.docx', '.txt', '.json'}
    files_to_delete = []
    
    for file_path in root_folder.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            files_to_delete.append(file_path)
    
    return sorted(files_to_delete)


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print("Error: No folder path provided")
        print(__doc__)
        sys.exit(1)
    
    folder_path = Path(sys.argv[1])
    dry_run = '--dry-run' in sys.argv
    skip_confirm = '--yes' in sys.argv or '-y' in sys.argv
    
    # Validate folder
    if not folder_path.exists():
        print(f"❌ Error: Folder does not exist: {folder_path}")
        sys.exit(1)
    
    if not folder_path.is_dir():
        print(f"❌ Error: Not a directory: {folder_path}")
        sys.exit(1)
    
    # Find files
    print(f"🔍 Scanning folder: {folder_path.absolute()}")
    files_to_delete = find_files_to_delete(folder_path)
    
    if not files_to_delete:
        print("✅ No .docx, .txt, or .json files found")
        return
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in files_to_delete if f.exists())
    
    # Group by extension for summary
    by_extension = {}
    for file in files_to_delete:
        ext = file.suffix.lower()
        by_extension[ext] = by_extension.get(ext, 0) + 1
    
    # Show summary
    print(f"\n📊 Found {len(files_to_delete)} file(s) to delete:")
    for ext, count in sorted(by_extension.items()):
        print(f"   {ext}: {count} file(s)")
    print(f"   Total size: {format_size(total_size)}")
    
    # Show files
    print("\n📄 Files to be deleted:")
    for file in files_to_delete:
        rel_path = file.relative_to(folder_path)
        size = format_size(file.stat().st_size) if file.exists() else "?"
        print(f"   {rel_path} ({size})")
    
    # Dry run mode
    if dry_run:
        print("\n🔍 DRY RUN - No files were deleted")
        return
    
    # Confirmation
    if not skip_confirm:
        print(f"\n⚠️  WARNING: This will permanently delete {len(files_to_delete)} file(s)")
        response = input("Are you sure you want to continue? [y/N]: ")
        if response.lower() not in ('y', 'yes'):
            print("❌ Cancelled")
            return
    
    # Delete files
    print("\n🗑️  Deleting files...")
    deleted_count = 0
    error_count = 0
    
    for file in files_to_delete:
        try:
            file.unlink()
            deleted_count += 1
            print(f"   ✓ {file.relative_to(folder_path)}")
        except Exception as e:
            error_count += 1
            print(f"   ✗ {file.relative_to(folder_path)}: {e}")
    
    # Summary
    print(f"\n✅ Deleted {deleted_count} file(s)")
    if error_count > 0:
        print(f"⚠️  Failed to delete {error_count} file(s)")
    print(f"💾 Freed up {format_size(total_size)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
