"""Example usage of Paper Fetchers"""

from src.fetchers import get_fetcher


def main():
    """Demonstrate fetcher usage"""

    print("=" * 60)
    print("PubMed Fetcher Example")
    print("=" * 60)

    # Get PubMed fetcher
    fetcher = get_fetcher("pubmed")

    print(f"\nUsing fetcher: {fetcher.__class__.__name__}")
    print(f"Base URL: {fetcher.base_url}")
    print(f"Rate limit: {fetcher.requests_per_second} req/sec")

    # Example 1: Search for papers
    print("\n" + "=" * 60)
    print("Example 1: Search for papers")
    print("=" * 60)

    query = "caloric restriction aging"
    print(f"\nSearching for: '{query}'")
    print("Fetching top 5 results...\n")

    pmids = fetcher.search(query, max_results=5)

    print(f"Found {len(pmids)} papers:")
    for i, pmid in enumerate(pmids, 1):
        print(f"{i}. PMID: {pmid}")

    # Example 2: Fetch paper metadata
    print("\n" + "=" * 60)
    print("Example 2: Fetch paper metadata")
    print("=" * 60)

    if pmids:
        pmid = pmids[0]
        print(f"\nFetching details for PMID: {pmid}...\n")

        paper = fetcher.fetch_paper(pmid)

        print(f"Title: {paper.title}")
        print(f"\nAuthors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
        print(f"Journal: {paper.journal}")
        print(f"Date: {paper.publication_date}")
        print(f"DOI: {paper.doi}")
        print(f"PMC ID: {paper.pmc_id}")
        print(f"\nAbstract (first 200 chars):")
        print(f"{paper.abstract[:200]}...")

        if paper.keywords:
            print(f"\nKeywords: {', '.join(paper.keywords)}")

    # Example 3: Search and fetch in one call
    print("\n" + "=" * 60)
    print("Example 3: Search and fetch multiple papers")
    print("=" * 60)

    query = "rapamycin longevity"
    print(f"\nSearching for: '{query}'")
    print("Fetching top 3 results with metadata...\n")

    papers = fetcher.search_and_fetch(query, max_results=3)

    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper.title}")
        print(f"   PMID: {paper.pmid} | Journal: {paper.journal}")
        print(f"   Authors: {paper.authors[0] if paper.authors else 'N/A'} et al.")
        print(f"   Date: {paper.publication_date}")

    # Example 4: Filter by publication date
    print("\n" + "=" * 60)
    print("Example 4: Search with date filter")
    print("=" * 60)

    # PubMed date filter syntax: YYYY/MM/DD
    query_with_date = "metformin aging AND 2020:2025[pdat]"
    print(f"\nSearching for: '{query_with_date}'")

    pmids = fetcher.search(query_with_date, max_results=5)
    print(f"Found {len(pmids)} papers from 2020-2025")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print("\nTips:")
    print("- Add your NCBI_API_KEY to .env for 10 req/sec (vs 3 without)")
    print("- Get API key: https://www.ncbi.nlm.nih.gov/account/settings/")
    print("- Use advanced search syntax: https://pubmed.ncbi.nlm.nih.gov/help/")


if __name__ == "__main__":
    main()
