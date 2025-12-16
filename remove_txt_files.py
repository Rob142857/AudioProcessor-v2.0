"""
Recursively Remove Text Files

One-off utility script to remove all .txt files from a folder and its subfolders.
Use with caution - deleted files cannot be recovered!
"""

import os
import sys


def remove_txt_files_recursive(root_folder: str, dry_run: bool = True):
    """
    Recursively remove all .txt files from the specified folder.
    
    Args:
        root_folder: Path to the root folder to search
        dry_run: If True, only shows what would be deleted without actually deleting
    """
    if not os.path.exists(root_folder):
        print(f"❌ Error: Folder does not exist: {root_folder}")
        return
    
    if not os.path.isdir(root_folder):
        print(f"❌ Error: Path is not a directory: {root_folder}")
        return
    
    print(f"🔍 Scanning: {root_folder}")
    print(f"{'🔍 DRY RUN MODE - No files will be deleted' if dry_run else '⚠️  DELETE MODE - Files will be permanently removed!'}")
    print("=" * 80)
    
    txt_files = []
    
    # Find all .txt files
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.txt'):
                full_path = os.path.join(dirpath, filename)
                txt_files.append(full_path)
    
    if not txt_files:
        print("✅ No .txt files found.")
        return
    
    print(f"📋 Found {len(txt_files)} .txt file(s):\n")
    
    deleted_count = 0
    failed_count = 0
    
    for txt_path in txt_files:
        rel_path = os.path.relpath(txt_path, root_folder)
        
        if dry_run:
            print(f"   Would delete: {rel_path}")
        else:
            try:
                os.remove(txt_path)
                print(f"   ✅ Deleted: {rel_path}")
                deleted_count += 1
            except Exception as e:
                print(f"   ❌ Failed: {rel_path} - {e}")
                failed_count += 1
    
    print("\n" + "=" * 80)
    if dry_run:
        print(f"🔍 DRY RUN: Would delete {len(txt_files)} file(s)")
        print("\nTo actually delete files, run with --confirm flag")
    else:
        print(f"✅ Successfully deleted: {deleted_count} file(s)")
        if failed_count > 0:
            print(f"❌ Failed to delete: {failed_count} file(s)")


if __name__ == "__main__":
    print("=" * 80)
    print("RECURSIVE .TXT FILE REMOVER")
    print("=" * 80)
    print()
    
    # Check for command line arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python remove_txt_files.py <folder_path>          (dry run - shows what would be deleted)")
        print("  python remove_txt_files.py <folder_path> --confirm (actually deletes files)")
        print()
        print("Example:")
        print('  python remove_txt_files.py "C:\\Users\\YourName\\Documents\\Transcripts"')
        print()
        
        # Interactive mode
        folder_path = input("Enter folder path (or press Enter to cancel): ").strip().strip('"')
        
        if not folder_path:
            print("Cancelled.")
            sys.exit(0)
    else:
        folder_path = sys.argv[1].strip().strip('"')
    
    # Check for confirm flag
    confirm = "--confirm" in sys.argv or "-y" in sys.argv
    
    if not confirm:
        print("\n⚠️  DRY RUN MODE - No files will be deleted")
        print("    This will show you what would be deleted")
        print()
    else:
        print("\n⚠️  WARNING: This will PERMANENTLY DELETE .txt files!")
        print(f"    From: {folder_path}")
        print()
        response = input("Are you absolutely sure? Type 'DELETE' to confirm: ")
        
        if response != "DELETE":
            print("Cancelled.")
            sys.exit(0)
        print()
    
    remove_txt_files_recursive(folder_path, dry_run=not confirm)
    
    print()
    print("Done!")
