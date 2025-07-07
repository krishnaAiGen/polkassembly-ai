import openai
import logging
from typing import List, Dict, Any, Optional

from .mem0_memory import get_memory_manager, add_user_query, add_assistant_response

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
                 web_search_context_size: str = "high",
                 enable_memory: bool = True):
        
        self.openai_api_key = openai_api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_web_search = enable_web_search
        self.web_search_context_size = web_search_context_size
        self.enable_memory = enable_memory
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.openai_api_key)
        
        # Initialize memory manager
        self.memory_manager = get_memory_manager() if enable_memory else None
        if self.memory_manager and self.memory_manager.enabled:
            logger.info("Memory functionality enabled")
        else:
            logger.info("Memory functionality disabled")
    
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
    
    def generate_answer_with_web_search(self, query: str, user_id: str = "default_user") -> Dict[str, Any]:
        """
        Generate answer using OpenAI with web search-like knowledge
        
        Args:
            query: User's question
            user_id: User identifier for memory operations
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            logger.info(f"Using web search fallback for query: '{query[:50]}...'")
            
            # Get memory context and add query to memory if memory is enabled
            memory_context = ""
            if self.memory_manager and self.memory_manager.enabled:
                memory_context = self.memory_manager.get_memory_context(query, user_id)
                self.memory_manager.add_user_query(query, user_id)
            
            # Create a comprehensive system prompt for web search-like responses
            system_prompt = """You are a helpful AI assistant specialized in answering questions about Polkadot, the blockchain platform. 

You have access to up-to-date knowledge about Polkadot and the broader blockchain ecosystem. When answering questions, draw from your knowledge of:
- Polkadot architecture and technology
- Parachains and the relay chain
- Governance and treasury systems
- Staking and validators
- Cross-chain interoperability
- Recent developments and upgrades
- Community and ecosystem projects

CRITICAL FORMATTING RULES - NEVER USE THESE SYMBOLS:
❌ FORBIDDEN: **, *, ##, ###, ->, •, ~~, `, ```, [], (), <>, |
❌ NO BOLD: **text** or __text__
❌ NO ITALICS: *text* or _text_
❌ NO HEADERS: # Header or ## Header
❌ NO CODE: `code` or ```code```
❌ NO LINKS: [text](url)

✅ REQUIRED FORMATTING:
- Use "1. 2. 3." for numbered lists only
- Use "- " (dash + space) for bullet points only
- Use regular sentences with periods
- Use paragraph breaks for separation
- Write prices as "110,011 USD" not "$110,011"
- Use quotation marks for emphasis: "important term"
- Keep responses CONCISE and PROFESSIONAL

EXAMPLE OF CORRECT FORMAT:
Polkadot governance operates through several key mechanisms:

1. Token holders can submit proposals for network changes
2. Voting uses conviction-based weighting system
3. Council members represent passive stakeholders
4. Technical committee handles emergency proposals
5. Treasury funds ecosystem development projects

The system balances community participation with technical expertise to ensure network security and growth.

IMPORTANT: Always follow these formatting rules exactly. Never use markdown formatting symbols. Write in clean, professional text suitable for chatbot responses."""
            
            # Create user prompt with memory context
            user_prompt_parts = []
            if memory_context:
                user_prompt_parts.append(f"Previous conversation context:\n{memory_context}")
            
            user_prompt_parts.append(f"Question about Polkadot: {query}")
            user_prompt = "\n\n".join(user_prompt_parts)
            
            # Generate response using OpenAI with enhanced model
            response = self.client.chat.completions.create(
                model="gpt-4o" if "gpt-4" in self.model or self.model == "gpt-3.5-turbo" else self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Lower temperature for more factual responses
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Clean up any markdown formatting that might have slipped through
            answer = self._clean_markdown_formatting(answer)
            
            # Add assistant response to memory if memory is enabled
            if self.memory_manager and self.memory_manager.enabled:
                self.memory_manager.add_assistant_response(answer, user_id)
            
            # Create web search-style sources
            sources = [
                {
                    'title': 'Polkadot Knowledge Base',
                    'url': 'https://polkadot.network',
                    'source_type': 'web_search',
                    'similarity_score': 0.9
                },
                {
                    'title': 'Polkadot Wiki',
                    'url': 'https://wiki.polkadot.network',
                    'source_type': 'web_search',
                    'similarity_score': 0.9
                },
                {
                    'title': 'Polkadot Documentation',
                    'url': 'https://docs.polkadot.network',
                    'source_type': 'web_search',
                    'similarity_score': 0.9
                }
            ]
            
            # Generate follow-up questions for web search results
            follow_up_questions = self._get_fallback_follow_ups(query)
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': 0.8,  # High confidence for knowledge-based responses
                'follow_up_questions': follow_up_questions,
                'context_used': True,
                'model_used': 'gpt-4o',
                'chunks_used': 0,
                'search_method': 'web_search'
            }
            
        except Exception as e:
            logger.error(f"Error with web search fallback: {e}")
            return {
                'answer': "I encountered an error while searching for information. Please try again or rephrase your question.",
                'sources': [
                    {
                        'title': 'Polkadot Network',
                        'url': 'https://polkadot.network',
                        'source_type': 'web_search',
                        'similarity_score': 1.0
                    }
                ],
                'confidence': 0.0,
                'follow_up_questions': self._get_fallback_follow_ups(query),
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
        introduction = """Hello! I'm the Polkassembly AI Assistant!

What is Polkassembly?
Polkassembly is the leading governance platform for Polkadot and Kusama ecosystems, designed to make blockchain governance accessible and transparent.

Key Features:
- Governance Tracking: Monitor proposals, referenda, and voting
- Democracy Tools: Participate in on-chain governance decisions  
- Analytics Dashboard: View governance statistics and trends
- Community Hub: Discuss proposals with other community members
- Voting Interface: Easy-to-use voting tools for token holders

Useful Links:
- Main Platform: https://polkassembly.io
- Polkadot Governance: https://polkadot.polkassembly.io
- Kusama Governance: https://kusama.polkassembly.io
- Documentation: https://docs.polkassembly.io
- GitHub: https://github.com/Premiurly/polkassembly

What can I help you with?
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

        # Generate greeting-specific follow-up questions
        greeting_follow_ups = [
            "How does Polkadot governance work?",
            "What are parachains and how do they work?",
            "How can I start staking DOT tokens?"
        ]
        
        return {
            'answer': introduction,
            'sources': sources,
            'confidence': 1.0,
            'follow_up_questions': greeting_follow_ups,
            'context_used': True,
            'model_used': 'polkassembly_intro',
            'chunks_used': 0,
            'search_method': 'greeting_response'
        }

    def generate_answer(self, 
                       query: str, 
                       chunks: List[Dict[str, Any]], 
                       custom_prompt: Optional[str] = None,
                       user_id: str = "default_user") -> Dict[str, Any]:
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
                
                # Handle memory for greeting
                if self.memory_manager and self.memory_manager.enabled:
                    self.memory_manager.add_user_query(query, user_id)
                
                greeting_response = self._get_polkassembly_introduction()
                
                # Add greeting response to memory
                if self.memory_manager and self.memory_manager.enabled:
                    self.memory_manager.add_assistant_response(greeting_response['answer'], user_id)
                
                return greeting_response
            
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
            
            # Get memory context if memory is enabled
            memory_context = ""
            if self.memory_manager and self.memory_manager.enabled:
                memory_context = self.memory_manager.get_memory_context(query, user_id)
                # Add user query to memory
                self.memory_manager.add_user_query(query, user_id)
            
            # Create system prompt
            system_prompt = custom_prompt or self._get_default_system_prompt()
            
            # Create user prompt with context, memory, and query
            user_prompt = self._create_user_prompt(query, context, memory_context)
            
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
            
            # Clean up any markdown formatting that might have slipped through
            answer = self._clean_markdown_formatting(answer)
            
            # Add assistant response to memory if memory is enabled
            if self.memory_manager and self.memory_manager.enabled:
                self.memory_manager.add_assistant_response(answer, user_id)
            
            # Extract sources from chunks
            sources = self._extract_sources(chunks)
            
            # Estimate confidence based on number of chunks and similarity scores
            confidence = self._estimate_confidence(chunks)
            
            # Generate follow-up questions
            follow_up_questions = self._generate_follow_up_questions(query, chunks, answer)
            
            result = {
                'answer': answer,
                'sources': sources,
                'confidence': confidence,
                'follow_up_questions': follow_up_questions,
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
                return self.generate_answer_with_web_search(query, user_id)
            
            return {
                'answer': "I encountered an error while generating the answer. Please try again.",
                'sources': [],
                'confidence': 0.0,
                'follow_up_questions': self._get_fallback_follow_ups(query),
                'context_used': False,
                'model_used': self.model,
                'chunks_used': 0,
                'search_method': 'error'
            }
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the QA system"""
        return """You are a helpful AI assistant specialized in answering questions about Polkadot, the blockchain platform. 

You will be provided with context from Polkadot documentation and forum posts. Please follow these guidelines:

1. Answer DIRECTLY - Do not mention sources, context, or documentation in your response
2. Provide CONCISE, SUMMARIZED responses - Keep answers brief but comprehensive
3. Never start with phrases like: "Based on the provided context", "According to the documentation", "From the Polkadot Wiki", etc.
4. Base your answers on the provided context but present them as direct knowledge
5. If the context doesn't contain enough information, simply state what you don't know without referencing the context
6. Be accurate and specific with relevant details
7. Explain technical concepts in a clear and understandable way
8. If you're uncertain about something, express that uncertainty directly
9. Structure your response professionally using clean text formatting
10. Avoid lengthy explanations - Focus on key points and essential information

CRITICAL FORMATTING RULES - NEVER USE THESE SYMBOLS:
❌ FORBIDDEN: **, *, ##, ###, ->, •, ~~, `, ```, [], (), <>, |
❌ NO BOLD: **text** or __text__
❌ NO ITALICS: *text* or _text_
❌ NO HEADERS: # Header or ## Header
❌ NO CODE: `code` or ```code```
❌ NO LINKS: [text](url)

✅ ALLOWED FORMATTING:
- Use "1. 2. 3." for numbered lists
- Use "- " (dash + space) for bullet points  
- Use regular sentences with periods
- Use paragraph breaks for separation
- Write prices as "110,011 USD" not "$110,011"
- Use quotation marks for emphasis: "important term"

EXAMPLE OF GOOD FORMATTING:
To stake DOT tokens:

1. Create and fund your wallet
2. Access Polkassembly website
3. Select reliable validators
4. Nominate your chosen validators
5. Monitor your staking rewards

Key benefits:
- Earn passive income through staking rewards
- Support network security and decentralization
- Participate in Polkadot governance decisions

Remember: Answer as if you have direct expertise about Polkadot. Provide helpful, accurate, and CONCISE information in clean, professional text format suitable for enterprise deployment."""
    
    def _create_user_prompt(self, query: str, context: str, memory_context: str = "") -> str:
        """Create the user prompt with query, context, and memory"""
        prompt_parts = []
        
        # Add memory context if available
        if memory_context:
            prompt_parts.append(f"Memory Context:\n{memory_context}")
        
        # Add document context
        if context:
            prompt_parts.append(f"Context Information:\n{context}")
        
        prompt_parts.append(f"Question: {query}")
        
        prompt_parts.append("Answer the question directly without mentioning the context, sources, documentation, or previous conversations. Do not start with phrases like \"Based on the provided context\", \"According to the documentation\", \"From the Polkadot Wiki\", \"From our previous conversation\", etc. Simply provide the answer as if you have direct knowledge of the topic.\n\nIMPORTANT: Use ONLY clean text formatting. NO markdown symbols like **, *, ##, •, etc. Use numbered lists (1. 2. 3.) and simple dashes (-) only.")
        
        return "\n\n".join(prompt_parts)
    
    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract source information from chunks, limited to top 2-3 most relevant"""
        sources = []
        seen_urls = set()
        
        # Sort chunks by similarity score and take top 3
        sorted_chunks = sorted(chunks, key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        for chunk in sorted_chunks[:3]:  # Limit to top 3 sources
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
            elif not source['url'] and len(sources) < 2:  # Only add non-URL sources if we have less than 2
                sources.append(source)
            
            # Stop if we have 2-3 good sources
            if len(sources) >= 2 and any(s['url'] for s in sources):
                break
        
        return sources[:3]  # Ensure maximum 3 sources
    
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
    
    def generate_summary(self, chunks: List[Dict[str, Any]]) -> str:
        """Generate a summary of the retrieved chunks"""
        try:
            if not chunks:
                return "No relevant information found."
            
            # Create a condensed context
            context = self.create_context_from_chunks(chunks, max_context_length=2000)
            
            summary_prompt = """Please provide a brief summary of the following Polkadot-related information:

{context}

Summary:"""
            
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": summary_prompt.format(context=context)}
                ],
                temperature=0.2,
                max_tokens=300
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Unable to generate summary."

    def _generate_follow_up_questions(self, query: str, chunks: List[Dict[str, Any]], answer: str) -> List[str]:
        """
        Generate 2-3 follow-up questions based on the query, context, and answer
        
        Args:
            query: Original user query
            chunks: Retrieved chunks that were used
            answer: Generated answer
            
        Returns:
            List of follow-up questions
        """
        try:
            # Extract key topics from chunks
            topics = set()
            for chunk in chunks[:3]:  # Only use top 3 chunks for topics
                metadata = chunk.get('metadata', {})
                title = metadata.get('title', '')
                content = chunk.get('content', '')
                
                # Extract potential topics from titles and content
                if title:
                    topics.add(title.split(' - ')[0])  # Get main topic before dash
                
                # Look for common Polkadot-related terms
                polkadot_terms = [
                    'parachain', 'relay chain', 'governance', 'staking', 'validator',
                    'nominator', 'treasury', 'referendum', 'proposal', 'council',
                    'DOT', 'KSM', 'kusama', 'democracy', 'xcm', 'bridge',
                    'consensus', 'runtime', 'substrate', 'crowdloan', 'auction'
                ]
                
                content_lower = content.lower()
                for term in polkadot_terms:
                    if term.lower() in content_lower and term.lower() not in query.lower():
                        topics.add(term)
            
            # Create context for follow-up generation
            topics_list = list(topics)[:5]  # Limit to 5 most relevant topics
            topics_str = ', '.join(topics_list) if topics_list else 'Polkadot ecosystem'
            
            # Generate follow-up questions using OpenAI
            follow_up_prompt = f"""Based on this query: "{query}"
And these related topics: {topics_str}

Generate exactly 3 short, relevant follow-up questions that a user might want to ask next. Each question should:
- Be directly related to the original query or mentioned topics
- Be concise (under 15 words)
- Explore different aspects or dive deeper
- Be practical and useful

Format: Return only the 3 questions, one per line, without numbers or bullets."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": follow_up_prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            follow_up_text = response.choices[0].message.content.strip()
            follow_up_questions = [q.strip() for q in follow_up_text.split('\n') if q.strip()]
            
            # Ensure we have 2-3 questions, fallback if needed
            if len(follow_up_questions) < 2:
                follow_up_questions = self._get_fallback_follow_ups(query)
            elif len(follow_up_questions) > 3:
                follow_up_questions = follow_up_questions[:3]
            
            return follow_up_questions
            
        except Exception as e:
            logger.warning(f"Error generating follow-up questions: {e}")
            return self._get_fallback_follow_ups(query)
    
    def _get_fallback_follow_ups(self, query: str) -> List[str]:
        """
        Get fallback follow-up questions based on common Polkadot topics
        
        Args:
            query: Original user query
            
        Returns:
            List of fallback follow-up questions
        """
        query_lower = query.lower()
        
        # Topic-based follow-ups
        if any(term in query_lower for term in ['governance', 'vote', 'proposal', 'referendum']):
            return [
                "How do I participate in Polkadot governance?",
                "What are the different governance tracks?",
                "How does voting power work in OpenGov?"
            ]
        elif any(term in query_lower for term in ['staking', 'validator', 'nominator']):
            return [
                "What are the risks of staking DOT?",
                "How do I choose good validators?",
                "What is the minimum amount needed to stake?"
            ]
        elif any(term in query_lower for term in ['parachain', 'slot', 'auction']):
            return [
                "How do parachain auctions work?",
                "What is the difference between parachains and parathreads?",
                "How do parachains communicate with each other?"
            ]
        elif any(term in query_lower for term in ['dot', 'token', 'price', 'economics']):
            return [
                "What are the main uses of DOT token?",
                "How does DOT inflation work?",
                "Where can I buy and store DOT?"
            ]
        else:
            # General follow-ups
            return [
                "How does Polkadot differ from other blockchains?",
                "What are the main benefits of using Polkadot?",
                "How can I get started with Polkadot?"
            ]
    
    def _clean_markdown_formatting(self, text: str) -> str:
        """
        Clean up markdown formatting from text to ensure clean chatbot responses
        
        Args:
            text: Input text that may contain markdown formatting
            
        Returns:
            Cleaned text without markdown formatting
        """
        import re
        
        # Remove bold formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'__(.*?)__', r'\1', text)
        
        # Remove italic formatting
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'_(.*?)_', r'\1', text)
        
        # Remove headers
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Remove code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove links but keep the text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove other markdown symbols
        text = re.sub(r'[•→~]', '-', text)
        text = re.sub(r'[<>|]', '', text)
        
        return text.strip() 