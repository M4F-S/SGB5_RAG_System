"""
SGB V RAG System - Web Scraper Module
Extracts German Social Code Book V (SGB V) sections from gesetze-im-internet.de
"""

import requests
from bs4 import BeautifulSoup
import json
import csv
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import urljoin
import urllib.robotparser
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class LegalSection:
    """Data class for a legal section"""
    section_id: str
    title: str
    text: str
    subsections: List[str]
    cross_references: List[str]
    hierarchy_level: int
    category: str
    last_updated: str
    full_text_with_metadata: str


class SGBVScraper:
    """
    Web scraper for German Social Code Book V (SGB V)
    
    Features:
    - Respects robots.txt and crawl delays
    - Handles nested paragraph structures (Absätze)
    - Preserves section hierarchy and cross-references
    - Error handling with retry logic
    - Rate limiting
    """
    
    BASE_URL = "https://www.gesetze-im-internet.de/sgb_5/"
    INDEX_URL = f"{BASE_URL}index.html"
    
    def __init__(
        self,
        delay_seconds: int = 2,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        respect_robots_txt: bool = True
    ):
        """
        Initialize scraper with configuration
        
        Args:
            delay_seconds: Delay between requests
            timeout_seconds: Request timeout
            max_retries: Number of retries on failure
            respect_robots_txt: Whether to respect robots.txt
        """
        self.delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.respect_robots_txt = respect_robots_txt
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set user agent
        self.session.headers.update({
            'User-Agent': 'SGB-V-RAG-System/1.0 (Educational; +https://github.com/yourusername/sgbv_rag_system)'
        })
        
        # Check robots.txt compliance
        if self.respect_robots_txt:
            self.robot_parser = urllib.robotparser.RobotFileParser()
            self.robot_parser.set_url(urljoin(self.BASE_URL, "robots.txt"))
            try:
                self.robot_parser.read()
                logger.info("robots.txt loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load robots.txt: {e}")
    
    def can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt"""
        if not self.respect_robots_txt:
            return True
        try:
            return self.robot_parser.can_fetch("*", url)
        except:
            return True
    
    def _fetch_page(self, url: str) -> Optional[str]:
        """
        Fetch a page with retry logic and rate limiting
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content or None if failed
        """
        if not self.can_fetch(url):
            logger.warning(f"Robots.txt blocks: {url}")
            return None
        
        time.sleep(self.delay_seconds)
        
        try:
            response = self.session.get(
                url,
                timeout=self.timeout_seconds
            )
            response.raise_for_status()
            logger.info(f"Successfully fetched: {url}")
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None
    
    def _extract_section_id(self, element) -> Optional[str]:
        """Extract section ID from HTML element"""
        # Try common patterns: §31, Paragraph 31, Section 31
        text = element.get_text(strip=True)
        
        # Pattern: §XXX or § XXX
        import re
        match = re.search(r'§\s*(\d+[a-z]?)', text)
        if match:
            return match.group(1)
        
        # Pattern: Paragraph XXX
        match = re.search(r'Paragraph\s+(\d+)', text)
        if match:
            return match.group(1)
        
        return None
    
    def _extract_cross_references(self, text: str) -> List[str]:
        """Extract cross-references (§XX) from text"""
        import re
        pattern = r'§\s*(\d+[a-z]?)'
        matches = re.findall(pattern, text)
        return list(set(matches))  # Remove duplicates
    
    def _parse_index_page(self, html: str) -> List[Tuple[str, str]]:
        """
        Parse index page to extract section links
        
        Returns:
            List of tuples (section_id, section_url)
        """
        soup = BeautifulSoup(html, 'html.parser')
        sections = []
        
        # Find all section links
        # SGB V structure uses <a> tags with section references
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            text = link.get_text(strip=True)
            
            # Match section patterns
            import re
            match = re.search(r'§?\s*(\d+[a-z]?)', text)
            if match and '.html' in href:
                section_id = match.group(1)
                full_url = urljoin(self.INDEX_URL, href)
                sections.append((section_id, full_url))
        
        logger.info(f"Found {len(sections)} sections in index")
        return sections
    
    def _parse_section_page(self, html: str, section_id: str) -> Optional[LegalSection]:
        """
        Parse a single section page
        
        Args:
            html: HTML content of section page
            section_id: Section identifier
            
        Returns:
            LegalSection object or None if parsing failed
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        try:
            # Extract title (usually in h1 or h2)
            title_elem = soup.find(['h1', 'h2'])
            title = title_elem.get_text(strip=True) if title_elem else f"Section {section_id}"
            
            # Extract main content
            # SGB sections are typically in <p> tags or <div> with specific classes
            content_divs = soup.find_all(['div', 'section'], class_=['toc', 'content', 'main'])
            text_content = []
            
            for div in content_divs:
                paragraphs = div.find_all('p')
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if text and len(text) > 10:
                        text_content.append(text)
            
            full_text = '\n'.join(text_content)
            
            if not full_text:
                # Fallback: get all text from body
                body = soup.find('body')
                full_text = body.get_text(separator='\n', strip=True) if body else ""
            
            # Extract subsections
            subsections = []
            import re
            subsection_pattern = r'§\s*(\d+[a-z]+)'
            subsections = list(set(re.findall(subsection_pattern, full_text)))
            
            # Extract cross-references
            cross_refs = self._extract_cross_references(full_text)
            
            return LegalSection(
                section_id=section_id,
                title=title,
                text=full_text,
                subsections=subsections,
                cross_references=cross_refs,
                hierarchy_level=1 if re.match(r'^\d+$', section_id) else 2,
                category="Health Insurance" if section_id.startswith('3') else "General",
                last_updated=datetime.now().isoformat(),
                full_text_with_metadata=f"Section {section_id}: {title}\n\n{full_text}"
            )
        
        except Exception as e:
            logger.error(f"Error parsing section {section_id}: {e}")
            return None
    
    def scrape_all(self) -> List[LegalSection]:
        """
        Scrape all SGB V sections
        
        Returns:
            List of LegalSection objects
        """
        logger.info("Starting SGB V scraping...")
        
        # Fetch index page
        index_html = self._fetch_page(self.INDEX_URL)
        if not index_html:
            logger.error("Failed to fetch index page")
            return []
        
        # Parse index to get section URLs
        sections = self._parse_index_page(index_html)
        
        # Scrape each section
        results = []
        for section_id, section_url in sections:
            logger.info(f"Scraping section {section_id}...")
            section_html = self._fetch_page(section_url)
            
            if section_html:
                section = self._parse_section_page(section_html, section_id)
                if section:
                    results.append(section)
            
            time.sleep(self.delay_seconds)
        
        logger.info(f"Scraping complete. Total sections: {len(results)}")
        return results
    
    def save_json(self, sections: List[LegalSection], output_path: str):
        """Save sections to JSON file"""
        data = [asdict(s) for s in sections]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(sections)} sections to {output_path}")
    
    def save_csv(self, sections: List[LegalSection], output_path: str):
        """Save sections to CSV file"""
        if not sections:
            logger.warning("No sections to save")
            return
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['section_id', 'title', 'hierarchy_level', 'category', 'last_updated']
            )
            writer.writeheader()
            
            for section in sections:
                writer.writerow({
                    'section_id': section.section_id,
                    'title': section.title,
                    'hierarchy_level': section.hierarchy_level,
                    'category': section.category,
                    'last_updated': section.last_updated
                })
        
        logger.info(f"Saved {len(sections)} sections to CSV {output_path}")


def main():
    """Main scraper execution"""
    import os
    
    # Create data directory
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Initialize scraper
    scraper = SGBVScraper(
        delay_seconds=2,
        timeout_seconds=30,
        max_retries=3,
        respect_robots_txt=True
    )
    
    # Scrape all sections
    sections = scraper.scrape_all()
    
    # Save results
    scraper.save_json(sections, 'data/raw/sgbv_sections.json')
    scraper.save_csv(sections, 'data/raw/sgbv_sections.csv')
    
    print(f"\n✓ Scraping complete: {len(sections)} sections")
    print(f"✓ JSON saved to: data/raw/sgbv_sections.json")
    print(f"✓ CSV saved to: data/raw/sgbv_sections.csv")


if __name__ == "__main__":
    main()
