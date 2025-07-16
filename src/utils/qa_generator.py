import openai
import logging
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QAGenerator:
    """Generate answers using OpenAI based on retrieved document chunks"""
    
    def __init__(self, 
                 openai_api_key: str,
                 model: str = "gpt-3.5-turbo",
                 temperature: float = 0.1,
                 max_tokens: int = 1000,
                 enable_web_search: bool = True,
                 web_search_context_size: str = "high"):
        
        self.openai_api_key = openai_api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_web_search = enable_web_search
        self.web_search_context_size = web_search_context_size
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.openai_api_key)
    
    def create_context_from_chunks(self, chunks: List[Dict[str, Any]], max_context_length: int = 4000) -> str:
        """
        Create context string from retrieved chunks
        
        Args:
            chunks: List of chunk dictionaries with content and metadata
            max_context_length: Maximum length of context in characters
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(chunks):
            # Format chunk with source information
            source_info = ""
            metadata = chunk.get('metadata', {})
            
            if metadata.get('title'):
                source_info += f"Title: {metadata['title']}\n"
            if metadata.get('url'):
                source_info += f"URL: {metadata['url']}\n"
            if metadata.get('source'):
                source_info += f"Source: {metadata['source']}\n"
            
            chunk_text = f"--- Document {i+1} ---\n{source_info}\nContent:\n{chunk['content']}\n\n"
            
            # Check if adding this chunk would exceed the max length
            if current_length + len(chunk_text) > max_context_length:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return ''.join(context_parts)
    
    def generate_answer_with_web_search(self, query: str) -> Dict[str, Any]:
        """
        Generate answer using OpenAI's web search capability via Responses API
        
        Args:
            query: User's question
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            logger.info(f"Using web search for query: '{query[:50]}...'")
            
            # Create a Polkadot-focused web search query
            web_search_query = f"Polkadot blockchain {query}"
            
            # Use OpenAI Responses API with web search tool
            tool = {
                "type": "web_search_preview", 
                "search_context_size": self.web_search_context_size
            }
            
            # Add user location for better search results
            user_location = {
                "type": "approximate",
                "country": "US",
                "city": "San Francisco",
                "timezone": "America/Los_Angeles"
            }
            tool["user_location"] = user_location
            
            # Add system instructions for summarized responses
            system_instruction = """You are a helpful AI assistant specialized in answering questions about Polkadot, the blockchain platform. 

Please provide CONCISE, SUMMARIZED responses that are:
- Brief but comprehensive
- Well-structured with bullet points or short paragraphs
- Focused on key points and essential information
- Technically accurate but easy to understand
- Avoid lengthy explanations - get straight to the point

Answer the following Polkadot-related question concisely:"""
            
            response = self.client.responses.create(
                model="gpt-4o",
                tools=[tool],
                input=f"{system_instruction}\n\n{web_search_query}"
            )
            
            answer = response.output_text.strip()
            
            # Extract citations if available
            sources = []
            try:
                if hasattr(response, 'output') and len(response.output) > 1:
                    annotations = response.output[1].content[0].annotations
                    if annotations:
                        for ann in annotations:
                            sources.append({
                                'title': ann.title,
                                'url': ann.url,
                                'source_type': 'web_search',
                                'similarity_score': 1.0
                            })
            except Exception as e:
                logger.warning(f"Could not extract citations: {e}")
                # Fallback source
                sources = [{'title': 'Web Search Results', 'url': '', 'source_type': 'web_search', 'similarity_score': 1.0}]
            
            if not sources:
                sources = [{'title': 'Web Search Results', 'url': '', 'source_type': 'web_search', 'similarity_score': 1.0}]
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': 0.8,  # High confidence for web search results
                'context_used': True,
                'model_used': 'gpt-4o',
                'chunks_used': 0,
                'search_method': 'web_search'
            }
            
        except Exception as e:
            logger.error(f"Error with web search: {e}")
            return {
                'answer': "I encountered an error while searching for information. Please try again or rephrase your question.",
                'sources': [],
                'confidence': 0.0,
                'context_used': False,
                'error': str(e),
                'search_method': 'web_search_failed'
            }
    
    def _is_greeting_message(self, query: str) -> bool:
        """
        Check if the query is a greeting or introduction message
        
        Args:
            query: User's question
            
        Returns:
            True if it's a greeting message
        """
        greeting_keywords = [
            'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 
            'good evening', 'what\'s up', 'whats up', 'wassup', 'howdy',
            'intro', 'introduction', 'who are you', 'what are you',
            'what do you do', 'what is this', 'help', 'start'
        ]
        
        query_lower = query.lower().strip()
        
        # Check for exact matches or if query starts with greeting
        for keyword in greeting_keywords:
            if query_lower == keyword or query_lower.startswith(keyword):
                return True
        
        # Check for short queries that might be greetings
        if len(query_lower.split()) <= 3:
            for keyword in greeting_keywords:
                if keyword in query_lower:
                    return True
        
        return False
    
    def _get_polkassembly_introduction(self) -> Dict[str, Any]:
        """
        Generate introduction response about Polkassembly
        
        Returns:
            Dictionary with introduction answer and metadata
        """
        introduction = """👋 **Hello! I'm the Polkassembly AI Assistant!**

**What is Polkassembly?**
Polkassembly is the leading governance platform for Polkadot and Kusama ecosystems, designed to make blockchain governance accessible and transparent.

**🔗 Key Features:**
• **Governance Tracking** - Monitor proposals, referenda, and voting
• **Democracy Tools** - Participate in on-chain governance decisions  
• **Analytics Dashboard** - View governance statistics and trends
• **Community Hub** - Discuss proposals with other community members
• **Voting Interface** - Easy-to-use voting tools for token holders

**🌐 Useful Links:**
• **Main Platform**: https://polkassembly.io
• **Polkadot Governance**: https://polkadot.polkassembly.io
• **Kusama Governance**: https://kusama.polkassembly.io
• **Documentation**: https://docs.polkassembly.io
• **GitHub**: https://github.com/Premiurly/polkassembly

**💡 What can I help you with?**
Ask me about Polkadot governance, parachains, staking, treasury proposals, referenda, or any other Polkadot/Kusama related topics!"""

        sources = [
            {
                'title': 'Polkassembly Main Platform',
                'url': 'https://polkassembly.io',
                'source_type': 'platform',
                'similarity_score': 1.0
            },
            {
                'title': 'Polkadot Governance on Polkassembly',
                'url': 'https://polkadot.polkassembly.io',
                'source_type': 'platform',
                'similarity_score': 1.0
            },
            {
                'title': 'Kusama Governance on Polkassembly',
                'url': 'https://kusama.polkassembly.io',
                'source_type': 'platform',
                'similarity_score': 1.0
            },
            {
                'title': 'Polkassembly Documentation',
                'url': 'https://docs.polkassembly.io',
                'source_type': 'documentation',
                'similarity_score': 1.0
            }
        ]

        return {
            'answer': introduction,
            'sources': sources,
            'confidence': 1.0,
            'context_used': True,
            'model_used': 'polkassembly_intro',
            'chunks_used': 0,
            'search_method': 'greeting_response'
        }

    def generate_answer(self, 
                       query: str, 
                       chunks: List[Dict[str, Any]], 
                       custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate an answer based on the query and retrieved chunks, with web search fallback
        
        Args:
            query: User's question
            chunks: Retrieved document chunks
            custom_prompt: Optional custom system prompt
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        try:
            # Check if this is a greeting message
            if self._is_greeting_message(query):
                logger.info("Detected greeting message, providing Polkassembly introduction")
                return self._get_polkassembly_introduction()
            
            # Create context from chunks
            context = self.create_context_from_chunks(chunks)
            
            # Check if we have sufficient context
            has_sufficient_context = (
                context.strip() and 
                len(chunks) > 0 and 
                any(chunk.get('similarity_score', 0) > 0.3 for chunk in chunks)
            )
            
            # If no sufficient context and web search is enabled, use web search
            if not has_sufficient_context and self.enable_web_search:
                logger.info("Insufficient local context, falling back to web search")
                return self.generate_answer_with_web_search(query)
            
            # If no context at all, return appropriate message
            if not context.strip():
                if self.enable_web_search:
                    # This shouldn't happen since we would have used web search above
                    return {
                        'answer': "I couldn't find relevant information in the Polkadot knowledge base to answer your question. Please try rephrasing your query or ask about a different topic related to Polkadot.",
                        'sources': [],
                        'confidence': 0.0,
                        'context_used': False,
                        'search_method': 'local_only'
                    }
                else:
                    return {
                        'answer': "I could not find sufficient information about the query.",
                        'sources': [],
                        'confidence': 0.0,
                        'context_used': False,
                        'search_method': 'local_only'
                    }
            
            # Create system prompt
            system_prompt = custom_prompt or self._get_default_system_prompt()
            
            # Create user prompt with context and query
            user_prompt = self._create_user_prompt(query, context)
            
            # Generate response using OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Extract sources from chunks
            sources = self._extract_sources(chunks)
            
            # Estimate confidence based on number of chunks and similarity scores
            confidence = self._estimate_confidence(chunks)
            
            result = {
                'answer': answer,
                'sources': sources,
                'confidence': confidence,
                'context_used': True,
                'model_used': self.model,
                'chunks_used': len(chunks),
                'search_method': 'local_knowledge'
            }
            
            logger.info(f"Generated answer for query: '{query[:50]}...' using {len(chunks)} chunks")
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            
            # If local generation fails and web search is enabled, try web search as fallback
            if self.enable_web_search:
                logger.info("Local generation failed, trying web search fallback")
                return self.generate_answer_with_web_search(query)
            
            return {
                'answer': "I encountered an error while generating the answer. Please try again.",
                'sources': [],
                'confidence': 0.0,
                'context_used': False,
                'error': str(e),
                'search_method': 'error'
            }
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the QA system"""
        return """You are a helpful AI assistant specialized in answering questions about Polkadot, the blockchain platform. 

You will be provided with context from Polkadot documentation and forum posts. Please follow these guidelines:

1. **Answer DIRECTLY** - Do not mention sources, context, or documentation in your response
2. **Provide CONCISE, SUMMARIZED responses** - Keep answers brief but comprehensive
3. **Never start with phrases like**: "Based on the provided context", "According to the documentation", "From the Polkadot Wiki", etc.
4. Base your answers on the provided context but present them as direct knowledge
5. If the context doesn't contain enough information, simply state what you don't know without referencing the context
6. Be accurate and specific with relevant details
7. Explain technical concepts in a clear and understandable way
8. If you're uncertain about something, express that uncertainty directly
9. Structure your response in a logical and easy-to-follow manner with bullet points or numbered lists when appropriate
10. **Avoid lengthy explanations** - Focus on key points and essential information
11. **Use clear, digestible formatting** with headers, bullet points, or short paragraphs

Remember: Answer as if you have direct expertise about Polkadot. Provide helpful, accurate, and CONCISE information while maintaining a natural, conversational tone."""
    
    def _create_user_prompt(self, query: str, context: str) -> str:
        """Create the user prompt with query and context"""
        return f"""Context Information:
{context}

Question: {query}

Answer the question directly without mentioning the context, sources, or documentation. Do not start with phrases like "Based on the provided context", "According to the documentation", "From the Polkadot Wiki", etc. Simply provide the answer as if you have direct knowledge of the topic."""
    
    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract source information from chunks"""
        sources = []
        seen_urls = set()
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            
            source = {
                'title': metadata.get('title', 'Unknown Title'),
                'url': metadata.get('url', ''),
                'source_type': metadata.get('source', 'unknown'),
                'similarity_score': chunk.get('similarity_score', 0.0)
            }
            
            # Avoid duplicate URLs
            if source['url'] and source['url'] not in seen_urls:
                sources.append(source)
                seen_urls.add(source['url'])
            elif not source['url']:
                sources.append(source)
        
        return sources
    
    def _estimate_confidence(self, chunks: List[Dict[str, Any]]) -> float:
        """Estimate confidence based on chunks and similarity scores"""
        if not chunks:
            return 0.0
        
        # Base confidence on average similarity score and number of chunks
        similarity_scores = [chunk.get('similarity_score', 0.0) for chunk in chunks]
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        
        # Adjust confidence based on number of chunks (more chunks = higher confidence, up to a point)
        chunk_bonus = min(len(chunks) * 0.1, 0.3)
        
        confidence = min(avg_similarity + chunk_bonus, 1.0)
        return round(confidence, 2)
    
 