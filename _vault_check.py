#!/usr/bin/env python3
"""Check entity index vs actual files."""
import re, os

vault_root = "/Users/darenpalmer/conductor/workspaces/psychohistory-v2/testbed/graph-vault"

# Read the entity index
with open(f"{vault_root}/entities/_index.md") as f:
    content = f.read()

# Extract all wikilinks
links = re.findall(r'\[\[([^\]]+)\]\]', content)

# Get actual entity files
entity_dir = f"{vault_root}/entities"
actual_files = set()
for fn in os.listdir(entity_dir):
    if fn.endswith('.md') and fn != '_index.md':
        actual_files.add(fn[:-3])

# Filter to entity-only links (no concepts/, events/, etc.)
entity_links = [l for l in links if '/' not in l and l not in ('_index', '_orchestrator_prerogatives')]

missing = set(entity_links) - actual_files
# Also check for listed historical figures that aren't actual files
# Lines with unlinked text like "Emilio Aguinaldo" etc.
listed_non_links = re.findall(r'- ([A-Z][a-z]+ [A-Z][a-z]+)', content)
listed_non_link_entities = []
for name in listed_non_links:
    slug = name.lower().replace(' ', '-')
    if slug not in actual_files and slug not in entity_links:
        listed_non_link_entities.append(name)

print(f"Entity links in _index.md: {len(entity_links)}")
print(f"Actual entity files: {len(actual_files)}")
print(f"Links with actual files: {len(set(entity_links) & actual_files)}")
print(f"Links WITHOUT files: {len(missing)}")
print()
if missing:
    print("=== MISSING (wikilinks with no file) ===")
    for m in sorted(missing):
        print(f"  - {m}")
print()
if listed_non_link_entities:
    print("=== LISTED BUT NOT WIKILINKED (plain text entries) ===")
    for m in listed_non_link_entities:
        print(f"  - {m}")

# Suggestion: which to remove from index vs which need files
print()
print("=== SUGGESTIONS ===")
for m in sorted(missing):
    print(f"  REMOVE from index: {m}")
