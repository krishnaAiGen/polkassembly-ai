#!/usr/bin/env python3
"""
Tests for enhanced AI-powered content guardrails.
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.content_guardrails import EnhancedContentGuardrails

class TestEnhancedGuardrails(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_api_key = "test-openai-key"
        self.guardrails = EnhancedContentGuardrails(self.test_api_key)
    
    def test_helpful_responses_categories(self):
        """Test that different categories return appropriate helpful responses"""
        # Test offensive content response
        response_offensive = self.guardrails._get_helpful_response('offensive_general')
        self.assertIn("help", response_offensive.lower())
        self.assertIn("polkadot", response_offensive.lower())
        
        # Test off-topic response
        response_off_topic = self.guardrails._get_helpful_response('off_topic')
        self.assertIn("polkadot", response_off_topic.lower())
        self.assertIn("blockchain", response_off_topic.lower())
        
        # Test inappropriate response
        response_inappropriate = self.guardrails._get_helpful_response('inappropriate')
        self.assertIn("polkadot", response_inappropriate.lower())
    
    def test_subsquare_filtering(self):
        """Test subsquare link filtering"""
        sources = [
            {'url': 'https://subsquare.io/polkadot/referendum/123', 'title': 'Subsquare Referendum'},
            {'url': 'https://polkadot.io/governance', 'title': 'Polkadot Governance'},
            {'url': 'https://polkassembly.io/referendum/123', 'title': 'Polkassembly Referendum'},
            {'url': 'https://subsquare.com/kusama/proposal/456', 'title': 'Another Subsquare'},
        ]
        
        filtered = self.guardrails.filter_sources(sources)
        
        # Should exclude all subsquare URLs
        self.assertEqual(len(filtered), 2)
        urls = [s['url'] for s in filtered]
        self.assertNotIn('subsquare.io', str(urls))
        self.assertNotIn('subsquare.com', str(urls))
        self.assertIn('polkadot.io', str(urls))
        self.assertIn('polkassembly.io', str(urls))
    
    def test_domain_prioritization(self):
        """Test that preferred domains are prioritized"""
        sources = [
            {'url': 'https://random-blog.com/polkadot', 'title': 'Random Blog', 'similarity_score': 0.9},
            {'url': 'https://polkadot.io/governance', 'title': 'Official Polkadot', 'similarity_score': 0.8},
            {'url': 'https://another-site.com/polkadot', 'title': 'Another Site', 'similarity_score': 0.85},
        ]
        
        filtered = self.guardrails.filter_sources(sources)
        
        # polkadot.io should be first due to domain prioritization
        self.assertEqual(filtered[0]['url'], 'https://polkadot.io/governance')
    
    def test_blocked_domain_detection(self):
        """Test blocked domain detection"""
        # Test various subsquare URLs
        test_urls = [
            'https://subsquare.io/polkadot',
            'https://www.subsquare.com/kusama',
            'https://subsquare.network/governance',
            'https://subsquare.org/referendum',
            'https://subsquare.app/proposal',
        ]
        
        for url in test_urls:
            self.assertTrue(self.guardrails._is_blocked_domain(url), f"Should block {url}")
        
        # Test allowed URLs
        allowed_urls = [
            'https://polkadot.io/governance',
            'https://polkassembly.io/referendum',
            'https://wiki.polkadot.network/docs',
        ]
        
        for url in allowed_urls:
            self.assertFalse(self.guardrails._is_blocked_domain(url), f"Should allow {url}")
    
    def test_response_sanitization(self):
        """Test response sanitization"""
        # Test subsquare reference removal
        response = "Check out subsquare.io for more info, it's super awesome stuff!"
        sanitized = self.guardrails.sanitize_response(response)
        
        # Should remove subsquare reference and improve tone
        self.assertNotIn('subsquare.io', sanitized)
        self.assertNotIn('super', sanitized)
        self.assertIn('excellent', sanitized)
        self.assertIn('features', sanitized)
    
    def test_professional_tone_enhancement(self):
        """Test professional tone enhancement"""
        casual_text = "This is super cool stuff that's basically awesome for guys to use"
        professional = self.guardrails._ensure_professional_tone(casual_text)
        
        # Check replacements
        self.assertNotIn('super', professional)
        self.assertNotIn('cool', professional)
        self.assertNotIn('stuff', professional)
        self.assertNotIn('basically', professional)
        self.assertNotIn('awesome', professional)
        self.assertNotIn('guys', professional)
        
        # Check professional alternatives
        self.assertIn('very', professional)
        self.assertIn('interesting', professional)
        self.assertIn('features', professional)
        self.assertIn('essentially', professional)
        self.assertIn('excellent', professional)
        self.assertIn('users', professional)
    
    def test_fallback_moderation(self):
        """Test regex fallback moderation"""
        # Test clean content
        is_safe, category, response = self.guardrails._fallback_moderation("How does Polkadot staking work?")
        self.assertTrue(is_safe)
        self.assertEqual(category, 'safe')
        
        # Test offensive content
        is_safe, category, response = self.guardrails._fallback_moderation("This is stupid and a scam")
        self.assertFalse(is_safe)
        self.assertEqual(category, 'offensive_general')
        self.assertIn("help", response.lower())
        self.assertIn("polkadot", response.lower())
    
    @patch('openai.OpenAI')
    def test_openai_moderation_integration(self, mock_openai_class):
        """Test OpenAI moderation API integration"""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Mock moderation response - safe content
        mock_moderation_result = MagicMock()
        mock_moderation_result.flagged = False
        mock_moderation_response = MagicMock()
        mock_moderation_response.results = [mock_moderation_result]
        mock_client.moderations.create.return_value = mock_moderation_response
        
        # Create guardrails with mocked client
        guardrails = EnhancedContentGuardrails("test-key")
        
        # Test safe content
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(guardrails._check_openai_moderation("How does Polkadot work?"))
        loop.close()
        
        self.assertTrue(result['is_safe'])
        self.assertEqual(result['category'], 'safe')
    
    @patch('openai.OpenAI')
    def test_openai_moderation_flagged_content(self, mock_openai_class):
        """Test OpenAI moderation API with flagged content"""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Mock moderation response - flagged content
        mock_moderation_result = MagicMock()
        mock_moderation_result.flagged = True
        mock_moderation_result.categories.__dict__ = {'hate': True, 'violence': False}
        mock_moderation_response = MagicMock()
        mock_moderation_response.results = [mock_moderation_result]
        mock_client.moderations.create.return_value = mock_moderation_response
        
        # Create guardrails with mocked client
        guardrails = EnhancedContentGuardrails("test-key")
        
        # Test flagged content
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(guardrails._check_openai_moderation("offensive content"))
        loop.close()
        
        self.assertFalse(result['is_safe'])
        self.assertEqual(result['category'], 'offensive_general')
        self.assertIn('hate', result['flagged_categories'])
    
    @patch('openai.OpenAI')
    def test_gpt_analysis_integration(self, mock_openai_class):
        """Test custom GPT analysis integration"""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Mock GPT analysis response
        mock_message = MagicMock()
        mock_message.content = '{"is_safe": true, "category": "safe", "reason": "Polkadot question", "is_polkadot_related": true}'
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_gpt_response = MagicMock()
        mock_gpt_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_gpt_response
        
        # Create guardrails with mocked client
        guardrails = EnhancedContentGuardrails("test-key")
        
        # Test GPT analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(guardrails._analyze_with_gpt("How does Polkadot work?"))
        loop.close()
        
        self.assertTrue(result['is_safe'])
        self.assertEqual(result['category'], 'safe')
        self.assertTrue(result['is_polkadot_related'])
    
    def test_synchronous_wrapper(self):
        """Test synchronous wrapper for async moderation"""
        # This will use fallback moderation since we don't have real API
        is_safe, category, response = self.guardrails.moderate_content("How does Polkadot work?")
        
        # Should be safe with fallback
        self.assertTrue(is_safe)
        self.assertEqual(category, 'safe')
    
    def test_empty_and_edge_cases(self):
        """Test empty and edge case inputs"""
        # Empty content
        self.assertEqual(self.guardrails.sanitize_response(""), "")
        self.assertEqual(self.guardrails.sanitize_response(None), "")
        
        # Empty URL checks
        self.assertFalse(self.guardrails._is_blocked_domain(""))
        self.assertFalse(self.guardrails._is_blocked_domain(None))
        
        # Invalid URL priority
        self.assertEqual(self.guardrails._get_domain_priority(""), 999)
        self.assertEqual(self.guardrails._get_domain_priority("invalid-url"), 999)

class TestGuardrailsIntegration(unittest.TestCase):
    """Integration tests for guardrails with the main system"""
    
    def test_get_guardrails_singleton(self):
        """Test that get_guardrails returns singleton instance"""
        from src.guardrail.content_guardrails import get_guardrails
        
        # Get two instances with same key
        guardrails1 = get_guardrails("test-key")
        guardrails2 = get_guardrails("test-key")
        
        # Should be the same instance
        self.assertIs(guardrails1, guardrails2)

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2) 