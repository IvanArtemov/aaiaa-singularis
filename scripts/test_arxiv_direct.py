"""
Quick test: Does arxiv.py library work?
"""

import arxiv

print("Testing arxiv.py library...")
print("=" * 60)

# Test 1: Search by ID (should always work)
print("\n[Test 1] Searching by ID: 1706.03762")
try:
    search = arxiv.Search(id_list=["1706.03762"])
    client = arxiv.Client()
    result = next(client.results(search))
    print(f"✓ SUCCESS: Found paper: {result.title}")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Test 2: Simple search query
print("\n[Test 2] Searching by query: 'quantum computing'")
try:
    search = arxiv.Search(
        query="quantum computing",
        max_results=3
    )
    client = arxiv.Client()
    results = list(client.results(search))
    print(f"✓ SUCCESS: Found {len(results)} papers")
    if results:
        print(f"  First paper: {results[0].title}")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Test 3: Category search
print("\n[Test 3] Searching by category: cat:cs.AI")
try:
    search = arxiv.Search(
        query="cat:cs.AI",
        max_results=3
    )
    client = arxiv.Client()
    results = list(client.results(search))
    print(f"✓ SUCCESS: Found {len(results)} papers")
    if results:
        print(f"  First paper: {results[0].title}")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Test 4: Our actual query
print("\n[Test 4] Our actual query: 'knowledge graph'")
try:
    search = arxiv.Search(
        query="knowledge graph",
        max_results=5
    )
    client = arxiv.Client()
    results = list(client.results(search))
    print(f"✓ SUCCESS: Found {len(results)} papers")
    if results:
        for i, r in enumerate(results[:3], 1):
            arxiv_id = r.entry_id.split('/')[-1].split('v')[0]
            print(f"  {i}. [{arxiv_id}] {r.title[:60]}...")
except Exception as e:
    print(f"✗ FAILED: {e}")

print("\n" + "=" * 60)
print("Testing complete!")