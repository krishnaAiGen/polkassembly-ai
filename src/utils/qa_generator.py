import openai
import logging
from typing import List, Dict, Any, Optional
import re
from .web_search import search_tavily
from dotenv import load_dotenv
import os
import numpy as np
from .gemini import GeminiClient
import time

from .mem0_memory import get_memory_manager, add_user_query, add_assistant_response
from ..guardrail.content_guardrails import get_guardrails

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

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
        
        # Initialize Gemini client (optional)
        try:
            self.gemini_client = GeminiClient(timeout=30)
            logger.info("Gemini client initialized successfully")
        except Exception as e:
            logger.warning(f"Gemini client initialization failed: {e}")
            logger.info("Continuing without Gemini client (OpenAI only mode)")
            self.gemini_client = None
        
        # Initialize enhanced guardrails
        self.guardrails = get_guardrails(openai_api_key)
        logger.info("Enhanced AI-powered guardrails initialized")
        
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
    
    async def generate_answer_with_web_search(self, query: str, user_id: str = "default_user") -> Dict[str, Any]:
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
            system_prompt = self._get_default_system_prompt()
            
            # Create user prompt with memory context
            user_prompt_parts = []
            if memory_context:
                user_prompt_parts.append(f"Previous conversation context:\n{memory_context}")
            
            user_prompt_parts.append(f"Question about Polkadot: {query}")
            user_prompt = "\n\n".join(user_prompt_parts)

            if os.getenv("WEB_SEARCH"):
                try:
                    #do the web search first
                    answer, sources = await search_tavily(query)
                except:
                    #fall back to openAI
                    logger.info("Web search failed, falling back to OpenAI")
                    response = self.client.chat.completions.create(
                        model="gpt-4o" if "gpt-4" in self.model or self.model == "gpt-3.5-turbo" else self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1,  # Lower temperature for more factual responses
                        max_tokens=self.max_tokens
                    )
                    
                    answer = response.choices[0].message.content
                    
                    # Clean any example.com URLs from the response
                    answer = self.clean_example_urls(answer)
                    
                    # answer = self.remove_double_asterisks(answer)
                    sources = []  # Initialize empty sources for fallback
            else:
                #no web search
                answer = "I couldn't find relevant information in the Polkadot knowledge base to answer your question based on web-search. Please try rephrasing your query or ask about a different topic related to Polkadot."
                sources = []
            
            # Apply content guardrails but preserve markdown formatting
            # answer = self.guardrails.sanitize_response(answer)
            
            # Add assistant response to memory if memory is enabled
            if self.memory_manager and self.memory_manager.enabled:
                self.memory_manager.add_assistant_response(answer, user_id)
            
            # Create web search-style sources
            if len(sources) > 0:
                sources = sources[:np.random.randint(1, 4)]
            else:
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
        introduction = """Hello! I'm the **Polkassembly AI Assistant**! 👋

**Polkassembly** is the leading governance platform for **Polkadot** and **Kusama** ecosystems, designed to make blockchain governance accessible and transparent.

**Key Features:**

- **Governance Tracking** - Monitor proposals, referenda, and voting activity

- **Democracy Tools** - Participate in on-chain governance decisions with easy voting interfaces

- **Analytics Dashboard** - View comprehensive governance analytics and voting patterns

- **Community Hub** - Discuss proposals and engage with other community members

**Useful Links:**

- **Main Platform**: [polkassembly.io](https://polkassembly.io)

- **Polkadot Governance**: [polkadot.polkassembly.io](https://polkadot.polkassembly.io)

- **Kusama Governance**: [kusama.polkassembly.io](https://kusama.polkassembly.io)

- **Documentation**: [docs.polkassembly.io](https://docs.polkassembly.io)

- **GitHub**: [github.com/Premiurly/polkassembly](https://github.com/Premiurly/polkassembly)

**What can I help you with?**

Ask me about **Polkadot governance**, *parachains*, *staking*, *treasury proposals*, *referenda*, or any other **Polkadot/Kusama** related topics! 🚀"""

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
    
    def remove_double_asterisks(self, text):
        return text.replace("**", "").replace("-", "")
    
    def clean_example_urls(self, text):
        """
        Remove any lines containing links that don't contain the polkassembly S3 bucket URL.
        Only allows links from https://polkassembly-ai.s3.us-east-1.amazonaws.com
        """
        import re
        
        lines = text.split('\n')
        cleaned_lines = []
        
        # Pattern to match image markdown: ![alt text](url)
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        
        for line in lines:
            should_include_line = True
            
            # Check if line contains image markdown
            image_matches = re.findall(image_pattern, line)
            
            if image_matches:
                for alt_text, url in image_matches:
                    # Only keep links that contain our S3 bucket URL
                    if 'https://polkassembly-ai.s3.us-east-1.amazonaws.com' not in url:
                        should_include_line = False
                        logger.info(f"Removed image line with invalid URL: {line.strip()}")
                        break
            
            if should_include_line:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    async def generate_answer(self, 
                       query: str, 
                       chunks: List[Dict[str, Any]], 
                       custom_prompt: Optional[str] = None,
                       user_id: str = "default_user") -> Dict[str, Any]:
        """
        Generate an answer based on the query and retrieved chunks, with web search fallback

        CRITICAL FORMATTING RULES:
        ✅ PROFESSIONAL FORMATTING REQUIREMENTS:
        - Use plain text without any markdown symbols (**, *, ##, -, etc.)
        - ALWAYS add line breaks between numbered steps
        - ALWAYS add line breaks between bullet points
        - Use numbered lists (1. 2. 3.) for step-by-step instructions with line breaks
        - Use simple bullet points without dashes or symbols
        - Write in clean, professional sentences
        - Use quotation marks for emphasis instead of bold/italic
        
        Args:
            query: User's question
            chunks: Retrieved document chunks
            custom_prompt: Optional custom system prompt
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        try:
            # AI-powered content moderation
            # is_safe, category, helpful_response = self.guardrails.moderate_content(query)
            
            # if not is_safe:
            #     logger.warning(f"Unsafe query detected - Category: {category}")
            #     return {
            #         'answer': helpful_response,
            #         'sources': [],
            #         'confidence': 0.0,
            #         'follow_up_questions': self._get_helpful_follow_ups(),
            #         'context_used': False,
            #         'model_used': self.model,
            #         'chunks_used': 0,
            #         'search_method': f'blocked_{category}'
            #     }
            
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
            
            # Create context from chunks (increased length limit for large chunks)
            context = self.create_context_from_chunks(chunks, max_context_length=8000)
            print("context from chunks", context)
            print("context after strip", context.strip())
            
            # Check if we have sufficient context
            has_sufficient_context = (
                context.strip() and 
                len(chunks) > 0 and 
                any(chunk.get('similarity_score', 0) > float(os.getenv("SIMILARITY_THRESHOLD")) for chunk in chunks)
            )

            print(f"similarity type is {type(os.getenv('SIMILARITY_THRESHOLD'))} and value if {float(os.getenv("SIMILARITY_THRESHOLD"))}")

            # If no sufficient context and web search is enabled, use web search
            # if not has_sufficient_context and self.enable_web_search:
            #     logger.info("Insufficient local context, falling back to web search")
            #     return await self.generate_answer_with_web_search(query)
            
            # If no sufficient context, diagnose the specific issue
            if not has_sufficient_context:
                max_score = max([chunk.get('similarity_score', 0) for chunk in chunks]) if chunks else 0
                has_good_similarity = any(chunk.get('similarity_score', 0) > 0.6 for chunk in chunks)
                has_chunks = len(chunks) > 0
                has_content = bool(context.strip())
                
                # Diagnose the specific problem
                if not has_chunks:
                    reason = "No relevant chunks found"
                elif not has_good_similarity:
                    reason = f"Low similarity scores (max: {max_score:.3f} < 0.6)"
                elif not has_content:
                    reason = f"Retrieved chunks contain no usable content (similarity: {max_score:.3f})"
                else:
                    reason = "Unknown context issue"
                
                logger.info(f"Insufficient context: {reason}. Query is outside knowledge boundary.")
                
                # Generate helpful follow-up questions for better guidance
                follow_up_questions = [
                    "How does Polkadot's governance system work?",
                    "What are parachains and how do they connect to Polkadot?",
                    "How can I stake DOT tokens on Polkadot?",
                    "What is the difference between Polkadot and Kusama?"
                ]
                
                return {
                    'answer': "I couldn't find sufficient information to answer your question accurately. This query appears to be outside my knowledge boundary for Polkadot-related topics. Please try rephrasing your question or ask about a more specific Polkadot topic.",
                    'sources': [],
                    'confidence': 0.0,
                    'context_used': False,
                    'model_used': self.model,
                    'chunks_used': len(chunks),
                    'follow_up_questions': follow_up_questions,
                    'search_method': 'insufficient_context',
                    'max_similarity_score': max_score,
                    'similarity_threshold': 0.6,
                    'failure_reason': reason
                }
            
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
                        'answer': "I could not find sufficient information about the query. Please try rephrasing your query or ask about a different topic related to Polkaseembly.",
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
            print("context before going to openAI prompt", context)
            user_prompt = self._create_user_prompt(query, context, memory_context)
            
            # Generate response using selected AI service (exclusive)
            answer = None
            openai_enabled = os.getenv("ENABLE_OPENAI", "").lower() == "true"
            gemini_enabled = os.getenv("ENABLE_GEMINI", "").lower() == "true"
            system_prompt = self._get_default_system_prompt()
            
            if openai_enabled:
                # Use OpenAI exclusively
                logger.info("Using OpenAI for response generation")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                answer = response.choices[0].message.content
                logger.info("OpenAI response received successfully")
                
            elif gemini_enabled and self.gemini_client:
                # Use Gemini exclusively  
                logger.info("Using Gemini for response generation")
                answer = self.gemini_client.get_response(system_prompt + "\n\n" + user_prompt)
                logger.info("Gemini response received successfully")
                
            else:
                # Fallback to OpenAI if no service is enabled
                logger.warning("No AI service explicitly enabled, falling back to OpenAI")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                answer = response.choices[0].message.content
                logger.info("OpenAI fallback response received successfully")
            print("--------answer without strip-------\n", answer)
            
            # Clean any example.com URLs from the response
            answer = self.clean_example_urls(answer)
            print("--------answer after cleaning example.com URLs-------\n", answer)
            
            # answer = self.remove_double_asterisks(answer)

            # Apply content guardrails but preserve markdown formatting
            # answer = self.guardrails.sanitize_response(answer)
            # print("-----answer after guardrail----\n", answer)
            
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
                return await self.generate_answer_with_web_search(query, user_id)
            
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

                ✅ PROFESSIONAL FORMATTING REQUIREMENTS:
                - ALWAYS add line breaks between numbered steps
                - ALWAYS add line breaks between bullet points
                - Use numbered lists (1. 2. 3.) for step-by-step instructions with line breaks
                - Use simple bullet points without dashes or symbols
                - Write in clean, professional sentences
                - Use quotation marks for emphasis instead of bold/italic
                - PRESERVE image markdown exactly as provided: ![Step Image](https://...) - keep this format unchanged
                - Include ALL images from the context in your response at the appropriate steps.
                - In your outuput, if there is any response related to Polkassembly, then nudge about polkassembly with its cool related features and links. 
                - If there is subsqure in your output, the omit any link related to subsquare in your output and nudge polkassembly.

            ## STEP-BY-STEP FORMATTING (MANDATORY):

            When providing numbered instructions, ALWAYS format like this:

            To stake DOT tokens and earn rewards:

            1. Create and fund your wallet with DOT tokens
               ![Step Image](https://example.com/image1.jpg)

            2. Access a staking interface (Polkadot.js, Polkassembly, etc.)
               ![Step Image](https://example.com/image2.jpg)

            3. Select reliable validators based on commission and performance

            4. Nominate your chosen validators with your desired amount

            5. Monitor your staking rewards and validator performance

            CRITICAL: Always include any images that appear in the context - they are essential visual guides!

            ## BULLET POINT FORMATTING:

            When listing features or benefits:

            Key benefits include:

            - Passive income** through staking rewards (typically 10-15% APY)
            - Network security** participation and decentralization support  
            - Governance rights** to vote on network proposals

            ## WHAT TO AVOID:

            NEVER format like this (bad example):
            "### How to stake DOT: 1. Create wallet 2. Select validators 3. Nominate tokens"
            "Never use https://example.com/image1.jpg or https://example.com/image2.jpg, in your output, if there is such, then remove it in the output"
            
            **Remember**: Answer as if you have direct expertise about Polkadot. Start directly with content, use proper line breaks between steps, and provide helpful, accurate, and **professionally formatted** information."""
                
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
        
        prompt_parts.append("Answer the question directly without mentioning the context, sources, documentation, or previous conversations. Do not start with phrases like \"Based on the provided context\", \"According to the documentation\", \"From the Polkadot Wiki\", \"From our previous conversation\", etc. Simply provide the answer as if you have direct knowledge of the topic.\n\nCRITICAL FORMATTING REQUIREMENTS:\n- NEVER start with headers (##, ###)\n- Start directly with answer content\n- ALWAYS add line breaks between numbered steps (1. step one [LINE BREAK] 2. step two [LINE BREAK])\n- ALWAYS add line breaks between bullet points\n- Use professional markdown formatting throughout\n- IMPORTANT: Include all images from the context using exact markdown format: ![Step Image](url)")
        
        return "\n\n".join(prompt_parts)
    
    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract and filter source information from chunks"""
        sources = []
        seen_urls = set()
        
        # Sort chunks by similarity score
        sorted_chunks = sorted(chunks, key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        for chunk in sorted_chunks[:3]:
            metadata = chunk.get('metadata', {})
            
            source = {
                'title': metadata.get('title', 'Unknown Title'),
                'url': metadata.get('url', ''),
                'source_type': metadata.get('source', 'unknown'),
                'similarity_score': chunk.get('similarity_score', 0.0)
            }
            
            if source['url'] and source['url'] not in seen_urls:
                sources.append(source)
                seen_urls.add(source['url'])
            elif not source['url'] and len(sources) < 2:
                sources.append(source)
            
            if len(sources) >= 2 and any(s['url'] for s in sources):
                break
        
        # Apply guardrails to filter sources (removes subsquare, prioritizes preferred)
        filtered_sources = self.guardrails.filter_sources(sources)
        
        return filtered_sources
    
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
            
            summary_prompt = """Please provide a brief summary of the following Polkadot-related information using proper markdown formatting.

IMPORTANT FORMATTING RULES:
- DO NOT start with headers (##, ###)
- Start directly with the summary content
- Use **bold** for key terms, *italics* for technical concepts
- Add line breaks between bullet points if used
- Keep it concise and professional

{context}

Summary:"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": summary_prompt.format(context=context)}
                ],
                temperature=0.2,
                max_tokens=300
            )
            
            summary = response.choices[0].message.content.strip()
            
            # Clean any example.com URLs from the summary
            summary = self.clean_example_urls(summary)
            
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
            
            # Clean any example.com URLs from the follow-up questions
            follow_up_text = self.clean_example_urls(follow_up_text)
            
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
    
    def _get_helpful_follow_ups(self) -> List[str]:
        """Get helpful follow-up questions that guide users to appropriate topics"""
        helpful_questions = [
            "How does Polkadot's governance system work?",
            "What are the benefits of staking DOT tokens?",
            "How do parachains communicate with each other?",
            "What makes Polkadot different from other blockchains?",
            "How can I participate in Polkadot governance?",
            "What are the risks and rewards of DOT staking?",
        ]
        
        import random
        return random.sample(helpful_questions, 3)

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