import logging
from typing import Any, Dict

from langchain_core.messages import AIMessage

from ...classes import ResearchState
from .base import BaseResearcher

logger = logging.getLogger(__name__)


class SocialMediaAnalyzer(BaseResearcher):
    def __init__(self) -> None:
        super().__init__()
        self.analyst_type = "social_media_analyzer"

    async def search_documents(self, state: ResearchState, queries: list[str]) -> Dict[str, Any]:
        """Override search_documents to include domain restrictions for social media sites"""
        websocket_manager = state.get('websocket_manager')
        job_id = state.get('job_id')

        if not queries:
            return {}

        # Send status update for generated queries
        if websocket_manager and job_id:
            await websocket_manager.send_status_update(
                job_id=job_id,
                status="queries_generated",
                message=f"Generated {len(queries)} queries for {self.analyst_type}",
                result={
                    "step": "Searching",
                    "analyst": self.analyst_type,
                    "queries": queries,
                    "total_queries": len(queries)
                }
            )

        # Social media domains to search
        social_media_domains = [
            "linkedin.com",
            "reddit.com", 
            "twitter.com",
            "x.com",
            "facebook.com",
            "instagram.com",
            "youtube.com",
            "tiktok.com"
        ]

        # Prepare search parameters with domain restrictions
        search_params = {
            "search_depth": "basic",
            "include_raw_content": False,
            "max_results": 5,
            "include_domains": social_media_domains
        }

        if websocket_manager and job_id:
            await websocket_manager.send_status_update(
                job_id=job_id,
                status="search_started",
                message=f"Searching social media platforms for {len(queries)} queries",
                result={
                    "step": "Searching",
                    "total_queries": len(queries),
                    "domains": social_media_domains
                }
            )

        # Execute searches in parallel
        import asyncio
        search_tasks = [
            self.tavily_client.search(query, **search_params)
            for query in queries
        ]

        try:
            results = await asyncio.gather(*search_tasks)
        except Exception as e:
            logger.error(f"Error during parallel social media search: {e}")
            return {}

        # Process results
        merged_docs = {}
        for query, result in zip(queries, results):
            for item in result.get("results", []):
                if not item.get("content") or not item.get("url"):
                    continue
                    
                url = item.get("url")
                title = item.get("title", "")
                
                # Import clean_title from utils
                from ...utils.references import clean_title
                if title:
                    title = clean_title(title)
                    if title.lower() == url.lower() or not title.strip():
                        title = ""

                merged_docs[url] = {
                    "title": title,
                    "content": item.get("content", ""),
                    "query": query,
                    "url": url,
                    "source": "social_media_search",
                    "score": item.get("score", 0.0)
                }

        # Send completion status
        if websocket_manager and job_id:
            await websocket_manager.send_status_update(
                job_id=job_id,
                status="search_complete",
                message=f"Social media search completed with {len(merged_docs)} documents found",
                result={
                    "step": "Searching",
                    "total_documents": len(merged_docs),
                    "queries_processed": len(queries)
                }
            )

        return merged_docs

    async def analyze(self, state: ResearchState) -> Dict[str, Any]:
        company = state.get('company', 'Unknown Company')
        msg = [f"📱 Social Media Analyzer analyzing {company}"]
        
        # Generate search queries using LLM
        queries = await self.generate_queries(state, """
        Generate queries on the social media presence and mentions of {company} such as:
        - Company LinkedIn profile and updates
        - Reddit discussions about the company
        - Twitter/X mentions and sentiment
        - Social media engagement and reputation
        """)

        # Add message to show subqueries with emojis
        subqueries_msg = "🔍 Subqueries for social media analysis:\n" + "\n".join([f"• {query}" for query in queries])
        messages = state.get('messages', [])
        messages.append(AIMessage(content=subqueries_msg))
        state['messages'] = messages

        # Send queries through WebSocket
        if websocket_manager := state.get('websocket_manager'):
            if job_id := state.get('job_id'):
                await websocket_manager.send_status_update(
                    job_id=job_id,
                    status="processing",
                    message="Social media analysis queries generated",
                    result={
                        "step": "Social Media Analyst",
                        "analyst_type": "Social Media Analyst",
                        "queries": queries
                    }
                )
        
        social_media_data = {}
        
        # If we have site_scrape data, include it first
        if site_scrape := state.get('site_scrape'):
            msg.append("\n📊 Including site scrape data in social media analysis...")
            company_url = state.get('company_url', 'company-website')
            social_media_data[company_url] = {
                'title': state.get('company', 'Unknown Company'),
                'raw_content': site_scrape,
                'query': f'Social media presence and mentions of {company}'
            }
        
        # Perform social media research with domain restrictions
        try:
            # Store documents with their respective queries
            for query in queries:
                documents = await self.search_documents(state, [query])
                if documents:  # Only process if we got results
                    for url, doc in documents.items():
                        doc['query'] = query  # Associate each document with its query
                        social_media_data[url] = doc
            
            msg.append(f"\n✓ Found {len(social_media_data)} social media documents")
            if websocket_manager := state.get('websocket_manager'):
                if job_id := state.get('job_id'):
                    await websocket_manager.send_status_update(
                        job_id=job_id,
                        status="processing",
                        message=f"Used Tavily Search to find {len(social_media_data)} social media documents",
                        result={
                            "step": "Searching",
                            "analyst_type": "Social Media Analyst",
                            "queries": queries
                        }
                    )
        except Exception as e:
            msg.append(f"\n⚠️ Error during social media research: {str(e)}")
        
        # Update state with our findings
        messages = state.get('messages', [])
        messages.append(AIMessage(content="\n".join(msg)))
        state['messages'] = messages
        state['social_media_data'] = social_media_data
        
        return {
            'message': msg,
            'social_media_data': social_media_data
        }

    async def run(self, state: ResearchState) -> Dict[str, Any]:
        return await self.analyze(state)