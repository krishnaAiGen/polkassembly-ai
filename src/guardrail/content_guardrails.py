#!/usr/bin/env python3
"""
Enhanced content guardrails using OpenAI moderation API and custom GPT analysis.
"""

import openai
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
import asyncio
import time
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedContentGuardrails:
    """Enhanced content filtering with AI-powered moderation"""
    
    def __init__(self, openai_api_key: str):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Blocked domains (subsquare and others)
        self.blocked_domains = [
            'subsquare.io',
            'subsquare.com', 
            'subsquare.network',
            'subsquare.org',
            'subsquare.app',
        ]
        
        # Preferred domains (prioritize these)
        self.preferred_domains = [
            'polkadot.io',
            'polkadot.network',
            'polkassembly.io',
            'wiki.polkadot.network',
            'docs.polkadot.network',
            'kusama.network',
            'github.com/paritytech',
            'medium.com/@polkadot',
            'polkadot.js.org',
        ]
        
        # Fallback regex patterns (backup to AI moderation)
        self.offensive_patterns = [
            r'\b(fuck|shit|damn|hell|crap|ass|bitch|bastard)\b',
            r'\b(stupid|idiot|moron|dumb|retard|loser)\b',
            r'\b(scam|fraud|ponzi|pyramid|fake|lie|lies)\b',
            r'\b(kill|die|death|murder|suicide)\b',
            r'\b(porn|sex|adult|explicit|nsfw)\b',
            r'\b(terrorist|extremist|radical|militant)\b',
        ]
        
        # Helpful redirect responses for different scenarios
        self.helpful_responses = {
            'offensive_general': [
                "I'm here to help you with Polkadot-related questions! How can I assist you with governance, staking, or technical features?",
                "Let me help you learn about Polkadot! Would you like to know about parachains, consensus mechanisms, or the DOT token?",
                "I'd be happy to help you with Polkadot topics! What would you like to know about - governance, staking, development, or something else?",
            ],
            'off_topic': [
                "I specialize in Polkadot and blockchain topics. How can I help you with governance, staking, parachain development, or ecosystem questions?",
                "I'm designed to help with Polkadot-related questions. Would you like to learn about DOT staking, governance participation, or parachain functionality?",
                "I focus on Polkadot ecosystem topics. Can I help you with validator selection, governance voting, or understanding parachains?",
            ],
            'inappropriate': [
                "I'm here to provide helpful information about Polkadot! What would you like to know about governance, staking, or the ecosystem?",
                "Let's focus on Polkadot topics where I can be most helpful! Are you interested in learning about staking rewards, governance, or parachains?",
                "I'd love to help you with Polkadot questions! Would you like to explore governance mechanisms, staking strategies, or development resources?",
            ]
        }
    
    async def moderate_content_with_ai(self, content: str) -> Tuple[bool, str, str]:
        """
        Use OpenAI's moderation API and custom GPT analysis for content filtering
        
        Args:
            content: Text content to moderate
            
        Returns:
            Tuple of (is_safe, category, helpful_response)
        """
        try:
            # First, try OpenAI's moderation API
            moderation_result = await self._check_openai_moderation(content)
            
            if not moderation_result['is_safe']:
                return False, moderation_result['category'], self._get_helpful_response('offensive_general')
            
            # Then use custom GPT analysis for context-aware filtering
            gpt_analysis = await self._analyze_with_gpt(content)
            
            return gpt_analysis['is_safe'], gpt_analysis['category'], gpt_analysis['helpful_response']
            
        except Exception as e:
            logger.error(f"AI moderation failed: {e}")
            # Fallback to regex-based filtering
            return self._fallback_moderation(content)
    
    async def _check_openai_moderation(self, content: str) -> Dict[str, Any]:
        """Use OpenAI's moderation API"""
        try:
            response = self.openai_client.moderations.create(input=content)
            result = response.results[0]
            
            if result.flagged:
                categories = [cat for cat, flagged in result.categories.__dict__.items() if flagged]
                return {
                    'is_safe': False,
                    'category': 'offensive_general',
                    'flagged_categories': categories
                }
            
            return {'is_safe': True, 'category': 'safe'}
            
        except Exception as e:
            logger.error(f"OpenAI moderation API error: {e}")
            raise e
    
    async def _analyze_with_gpt(self, content: str) -> Dict[str, Any]:
        """Use custom GPT prompt for context-aware content analysis"""
        try:
            analysis_prompt = f"""
You are a content moderator for a Polkadot blockchain educational chatbot. Analyze this user input and determine:

1. Is it safe and appropriate for a professional blockchain discussion?
2. Is it related to Polkadot/blockchain topics?
3. What category does it fall into?

User input: "{content}"

Respond with ONLY a JSON object:
{{
    "is_safe": true/false,
    "category": "safe" | "offensive_general" | "off_topic" | "inappropriate",
    "reason": "brief explanation",
    "is_polkadot_related": true/false
}}

Categories:
- "safe": Clean, appropriate, and potentially Polkadot-related
- "offensive_general": Contains offensive language or inappropriate content
- "off_topic": Not related to Polkadot/blockchain but not offensive
- "inappropriate": Inappropriate but not necessarily offensive

Be helpful and educational - err on the side of being permissive for genuine questions.
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=150
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                analysis = json.loads(result_text)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                analysis = {
                    'is_safe': True,
                    'category': 'safe',
                    'reason': 'Analysis parsing failed',
                    'is_polkadot_related': False
                }
            
            # Generate helpful response based on category
            helpful_response = self._get_helpful_response(analysis['category'])
            
            return {
                'is_safe': analysis['is_safe'],
                'category': analysis['category'],
                'helpful_response': helpful_response,
                'reason': analysis.get('reason', ''),
                'is_polkadot_related': analysis.get('is_polkadot_related', False)
            }
            
        except Exception as e:
            logger.error(f"GPT analysis error: {e}")
            return {
                'is_safe': True,
                'category': 'safe',
                'helpful_response': self._get_helpful_response('safe'),
                'reason': 'Analysis failed'
            }
    
    def _fallback_moderation(self, content: str) -> Tuple[bool, str, str]:
        """Fallback regex-based moderation"""
        content_lower = content.lower()
        
        for pattern in self.offensive_patterns:
            if re.search(pattern, content_lower, re.IGNORECASE):
                logger.info(f"Fallback filter triggered: {pattern}")
                return False, 'offensive_general', self._get_helpful_response('offensive_general')
        
        return True, 'safe', ''
    
    def _get_helpful_response(self, category: str) -> str:
        """Get a helpful redirect response based on category"""
        import random
        
        if category in self.helpful_responses:
            return random.choice(self.helpful_responses[category])
        else:
            return random.choice(self.helpful_responses['offensive_general'])
    
    def filter_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter sources to exclude blocked domains and prioritize preferred ones"""
        filtered_sources = []
        
        for source in sources:
            url = source.get('url', '')
            
            # Skip sources with blocked domains
            if self._is_blocked_domain(url):
                logger.info(f"Blocked source: {url}")
                continue
            
            # Skip sources with no URL (but allow some if we have few sources)
            if not url or url.strip() == '':
                if len(filtered_sources) < 2:
                    filtered_sources.append(source)
                continue
            
            filtered_sources.append(source)
        
        # Sort by preferred domains first
        filtered_sources.sort(key=lambda x: self._get_domain_priority(x.get('url', '')))
        
        return filtered_sources[:3]  # Limit to top 3 sources
    
    def _is_blocked_domain(self, url: str) -> bool:
        """Check if URL contains blocked domain"""
        if not url:
            return False
            
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower().replace('www.', '')
            
            return any(blocked in domain for blocked in self.blocked_domains)
        except:
            return False
    
    def _get_domain_priority(self, url: str) -> int:
        """Get priority score for URL (lower is better)"""
        if not url:
            return 999
            
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower().replace('www.', '')
            
            # Check if it's a preferred domain
            for i, preferred in enumerate(self.preferred_domains):
                if preferred in domain:
                    return i
            
            return 500  # Default priority
        except:
            return 999
    
    def sanitize_response(self, response: str) -> str:
        """Sanitize AI response to ensure it's appropriate"""
        if not response:
            return ""
        
        # Remove any subsquare references
        response = self._remove_blocked_references(response)
        
        # Ensure professional tone
        response = self._ensure_professional_tone(response)
        
        return response
    
    def _remove_blocked_references(self, content: str) -> str:
        """Remove references to blocked domains"""
        patterns = [
            r'subsquare\.io[^\s]*',
            r'subsquare\.com[^\s]*', 
            r'subsquare\.network[^\s]*',
            r'subsquare\.org[^\s]*',
            r'subsquare\.app[^\s]*',
            r'\bsubsquare\b',
        ]
        
        for pattern in patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content
    
    def _ensure_professional_tone(self, content: str) -> str:
        """Ensure content maintains professional tone"""
        replacements = {
            r'\bawesome\b': 'excellent',
            r'\bsuper\b': 'very',
            r'\bcool\b': 'interesting',
            r'\bstuff\b': 'features',
            r'\bguys\b': 'users',
            r'\bthing\b': 'feature',
            r'\bbasically\b': 'essentially',
        }
        
        for pattern, replacement in replacements.items():
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        return content
    
    # Synchronous wrapper for async methods
    def moderate_content(self, content: str) -> Tuple[bool, str, str]:
        """
        Synchronous wrapper for content moderation
        """
        try:
            # Create a new event loop for this thread if one doesn't exist
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async moderation in the event loop
            return loop.run_until_complete(self.moderate_content_with_ai(content))
        except Exception as e:
            logger.error(f"Content moderation error: {e}")
            # Fallback to regex-based filtering on error
            return self._fallback_moderation(content)

# Global instance
_guardrails_instance = None

def get_guardrails(openai_api_key: str) -> EnhancedContentGuardrails:
    """Get the global guardrails instance"""
    global _guardrails_instance
    if _guardrails_instance is None:
        _guardrails_instance = EnhancedContentGuardrails(openai_api_key)
    return _guardrails_instance 