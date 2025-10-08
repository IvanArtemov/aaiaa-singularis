"""PubMed fetcher using NCBI E-utilities API"""

import requests
import xml.etree.ElementTree as ET
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from .base_fetcher import BaseFetcher, PaperMetadata


class PubMedFetcher(BaseFetcher):
    """Fetcher for PubMed using E-utilities API"""

    def __init__(self, config: Dict[str, Any], api_key: str = "", tool_name: str = "", email: str = ""):
        super().__init__(config)
        self.api_key = api_key
        self.database = config.get("database", "pubmed")
        self.endpoints = config.get("endpoints", {})
        self.requests_per_second = config.get("rate_limit", {}).get("requests_per_second", 3)
        self.delay_between_requests = 1.0 / self.requests_per_second
        self.last_request_time = 0

        # REQUIRED: Tool name and email to avoid IP blocking
        self.tool_name = tool_name or os.getenv("NCBI_TOOL_NAME", "")
        self.email = email or os.getenv("NCBI_EMAIL", "")

        # Warn if missing required parameters
        if not self.tool_name or not self.email:
            print("WARNING: NCBI_TOOL_NAME and NCBI_EMAIL not set!")
            print("Your IP may be blocked. Set them in .env file.")
            print("See: https://www.ncbi.nlm.nih.gov/books/NBK25497/")

    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.delay_between_requests:
            time.sleep(self.delay_between_requests - time_since_last_request)
        self.last_request_time = time.time()

    def _build_url(self, endpoint: str, **params) -> str:
        """Build E-utilities URL"""
        endpoint_path = self.endpoints.get(endpoint, endpoint)
        url = f"{self.base_url}/{endpoint_path}"

        # Add database and API key
        params["db"] = self.database
        if self.api_key:
            params["api_key"] = self.api_key

        # REQUIRED: Add tool and email to avoid IP blocking
        if self.tool_name:
            params["tool"] = self.tool_name
        if self.email:
            params["email"] = self.email

        # Build query string
        query_parts = [f"{key}={value}" for key, value in params.items()]
        return f"{url}?{'&'.join(query_parts)}"

    def search(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",
        free_full_text: bool = False,
        **kwargs
    ) -> List[str]:
        """
        Search PubMed using ESearch

        Args:
            query: Search query (e.g., "caloric restriction aging")
            max_results: Maximum number of results
            sort_by: Sort order ("relevance", "pub_date", etc.)
            free_full_text: If True, filter to only free full text articles
            **kwargs: Additional E-utilities parameters

        Returns:
            List of PMIDs
        """
        self._rate_limit()

        # Add free full text filter if requested
        search_term = query
        if free_full_text:
            search_term = f"{query} AND free full text[filter]"

        url = self._build_url(
            "search",
            term=search_term,
            retmax=max_results,
            sort=sort_by,
            retmode="json",
            **kwargs
        )

        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        id_list = data.get("esearchresult", {}).get("idlist", [])

        return id_list

    def fetch_paper(self, paper_id: str) -> PaperMetadata:
        """
        Fetch paper metadata using EFetch

        Args:
            paper_id: PubMed ID (PMID)

        Returns:
            PaperMetadata: Paper information
        """
        self._rate_limit()

        url = self._build_url(
            "fetch",
            id=paper_id,
            retmode="xml",
            rettype="abstract"
        )

        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()

        # Parse XML response
        root = ET.fromstring(response.content)
        article = root.find(".//PubmedArticle")

        if article is None:
            raise ValueError(f"Paper {paper_id} not found")

        return self._parse_article_xml(article, paper_id)

    def _parse_article_xml(self, article: ET.Element, pmid: str) -> PaperMetadata:
        """Parse PubMed article XML"""

        # Title
        title_elem = article.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None else ""

        # Authors
        authors = []
        for author in article.findall(".//Author"):
            last_name = author.find("LastName")
            fore_name = author.find("ForeName")
            if last_name is not None and fore_name is not None:
                authors.append(f"{fore_name.text} {last_name.text}")
            elif last_name is not None:
                authors.append(last_name.text)

        # Abstract
        abstract_parts = []
        for abstract_text in article.findall(".//AbstractText"):
            if abstract_text.text:
                # Include label if exists
                label = abstract_text.get("Label", "")
                text = abstract_text.text
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
        abstract = "\n".join(abstract_parts)

        # Journal
        journal_elem = article.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else ""

        # Publication date
        pub_date_elem = article.find(".//PubDate")
        pub_date = ""
        if pub_date_elem is not None:
            year = pub_date_elem.find("Year")
            month = pub_date_elem.find("Month")
            day = pub_date_elem.find("Day")
            date_parts = []
            if year is not None:
                date_parts.append(year.text)
            if month is not None:
                date_parts.append(month.text)
            if day is not None:
                date_parts.append(day.text)
            pub_date = "-".join(date_parts)

        # Keywords
        keywords = []
        for keyword in article.findall(".//Keyword"):
            if keyword.text:
                keywords.append(keyword.text)

        # DOI
        doi = None
        for article_id in article.findall(".//ArticleId"):
            if article_id.get("IdType") == "doi":
                doi = article_id.text
                break

        # PMC ID
        pmc_id = None
        for article_id in article.findall(".//ArticleId"):
            if article_id.get("IdType") == "pmc":
                pmc_id = article_id.text
                break

        # Determine if free full text is available (indicated by PMC ID)
        has_free_full_text = pmc_id is not None

        return PaperMetadata(
            pmid=pmid,
            pmc_id=pmc_id,
            doi=doi,
            title=title,
            authors=authors,
            abstract=abstract,
            journal=journal,
            publication_date=pub_date,
            keywords=keywords,
            has_free_full_text=has_free_full_text
        )

    def fetch_full_text(self, paper_id: str) -> Optional[str]:
        """
        Fetch full text from PMC (if available)

        Note: This requires the paper to be in PMC Open Access Subset
        Use ELink to find PMC ID first

        Args:
            paper_id: PMID

        Returns:
            Full text or None if not available
        """
        # First, try to get PMC ID via ELink
        pmc_id = self._get_pmc_id(paper_id)

        if not pmc_id:
            return None

        # Fetch full text from PMC
        # Note: This is a simplified version
        # Full implementation would use PMC OAI or BioC API
        return None  # Placeholder for now

    def _get_pmc_id(self, pmid: str) -> Optional[str]:
        """Get PMC ID from PMID using ELink"""
        self._rate_limit()

        url = self._build_url(
            "link",
            dbfrom="pubmed",
            db="pmc",
            id=pmid,
            retmode="json"
        )

        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            linksets = data.get("linksets", [])
            if linksets and len(linksets) > 0:
                linksetdbs = linksets[0].get("linksetdbs", [])
                for linksetdb in linksetdbs:
                    if linksetdb.get("linkname") == "pubmed_pmc":
                        links = linksetdb.get("links", [])
                        if links:
                            # Add PMC prefix if not present
                            pmc_id = str(links[0])
                            if not pmc_id.startswith("PMC"):
                                pmc_id = f"PMC{pmc_id}"
                            return pmc_id
        except Exception as e:
            print(f"Error getting PMC ID: {e}")

        return None

    def search_and_fetch(
        self,
        query: str,
        max_results: int = 10
    ) -> List[PaperMetadata]:
        """
        Convenience method: search and fetch papers in one call

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of PaperMetadata
        """
        pmids = self.search(query, max_results=max_results)
        papers = self.fetch_multiple(pmids)
        return papers

    def download_pdf(self, paper_id: str, output_dir: str = "articles/packages") -> Optional[str]:
        """
        Download tar.gz package from PubMed Central OA Service

        Args:
            paper_id: PMID or PMC ID
            output_dir: Directory to save tar.gz package

        Returns:
            Path to downloaded tar.gz file or None if not available
        """
        self._rate_limit()

        # Get PMC ID if we have PMID
        pmc_id = None
        if paper_id.startswith("PMC"):
            pmc_id = paper_id
            print(f"Using provided PMC ID: {pmc_id}")
        else:
            # Try to get PMC ID from PMID
            print(f"Fetching PMC ID for PMID {paper_id}...")
            pmc_id = self._get_pmc_id(paper_id)
            if pmc_id:
                print(f"  Found PMC ID: {pmc_id}")

        if not pmc_id:
            print(f"✗ No PMC ID found for {paper_id} - package not available")
            return None

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Use PMC OA Service to get tar.gz package
            print(f"Accessing PMC OA Service...")
            oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmc_id}"
            response = requests.get(oa_url, timeout=self.timeout)
            response.raise_for_status()

            # Parse XML response
            root = ET.fromstring(response.content)
            record = root.find(".//record")

            if record is None:
                print(f"✗ No OA record found for {pmc_id}")
                return None

            # Get link to tar.gz package
            link = record.find("link[@format='tgz']")
            if link is None:
                print(f"✗ No tar.gz package available for {pmc_id}")
                return None

            package_url = link.get('href')
            print(f"  Found package URL: {package_url}")

            # Convert FTP to HTTPS (requests doesn't support FTP)
            if package_url.startswith('ftp://'):
                package_url = package_url.replace('ftp://', 'https://')
                print(f"  Using HTTPS: {package_url}")

            # Download tar.gz package
            print(f"  Downloading tar.gz package...")
            self._rate_limit()  # Rate limit before download

            package_response = requests.get(package_url, timeout=60, stream=True)
            package_response.raise_for_status()

            # Save tar.gz file
            filename = f"{pmc_id}.tar.gz"
            output_file = output_path / filename

            with open(output_file, 'wb') as f:
                for chunk in package_response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = output_file.stat().st_size
            print(f"✓ Downloaded tar.gz package to {output_file} ({file_size / 1024:.1f} KB)")

            return str(output_file)

        except Exception as e:
            print(f"✗ Download failed: {e}")
            return None
