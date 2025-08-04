import os
import signal
from contextlib import contextmanager
from dotenv import load_dotenv
from google import genai

@contextmanager
def timeout_context(seconds):
    """Context manager for timeout handling"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Reset the alarm and restore the old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

class GeminiClient:
    """A client class for interacting with Google's Gemini 2.5 Flash model."""
    
    def __init__(self, model_name="gemini-2.5-flash-lite", timeout=30):
        """
        Initialize the Gemini client.
        
        Args:
            model_name (str): The name of the Gemini model to use
            timeout (int): Timeout for API calls in seconds
        """
        # Load environment variables from .env file
        load_dotenv()
        
        # Get API key from environment
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set. Please check your .env file.")
        
        self.timeout = timeout
        self.model_name = model_name
        
        # Test API key validity during initialization
        print(f"Initializing Gemini client with model: {model_name}")
        print(f"API Key (first 10 chars): {self.api_key[:10]}...")
        
        try:
            # Initialize the client
            self.client = genai.Client(api_key=self.api_key)
            print("✅ Client initialized successfully")
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini client: {str(e)}")
        
    def get_response(self, prompt, timeout=None, **kwargs):
        """
        Generate a response from the Gemini model with timeout handling.
        
        Args:
            prompt (str): The input prompt for the model
            timeout (int): Override default timeout for this request
            **kwargs: Additional generation configuration options
            
        Returns:
            str: The generated response text
        """
        request_timeout = timeout or self.timeout
        
        try:
            print(f"🤖 Sending request to {self.model_name}...")
            print(f"⏱️  Timeout set to: {request_timeout} seconds")
            print(f"📝 Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
            
            with timeout_context(request_timeout):
                # Generate response using the correct API format
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    **kwargs
                )
                
                print("✅ Response received successfully")
                return response.text
                
        except TimeoutError as e:
            error_msg = f"Request timed out after {request_timeout} seconds. Try increasing timeout or check your internet connection."
            print(f"⏰ {error_msg}")
            return error_msg
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Provide more specific error guidance
            if "api_key" in str(e).lower():
                error_msg += "\n💡 Check if your GEMINI_API_KEY is valid and has proper permissions."
            elif "quota" in str(e).lower() or "limit" in str(e).lower():
                error_msg += "\n💡 You may have exceeded your API quota or rate limits."
            elif "network" in str(e).lower() or "connection" in str(e).lower():
                error_msg += "\n💡 Check your internet connection and try again."
            
            return error_msg

    def chat(self, messages, timeout=None, **kwargs):
        """
        Have a conversation with the Gemini model.
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            timeout (int): Override default timeout for this request
            **kwargs: Additional generation configuration options
            
        Returns:
            str: The generated response text
        """
        try:
            # Convert messages to the format expected by Gemini
            contents = []
            for msg in messages:
                contents.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}")
            
            conversation = "\n".join(contents)
            print(f"💬 Starting chat conversation with {len(messages)} messages")
            
            return self.get_response(conversation, timeout=timeout, **kwargs)
            
        except Exception as e:
            return f"Error in chat: {str(e)}"

    def test_connection(self):
        """Test the connection with a simple request"""
        print("🔍 Testing connection...")
        try:
            test_response = self.get_response("Hello", timeout=10)
            if "error" not in test_response.lower():
                print("✅ Connection test successful!")
                return True
            else:
                print(f"❌ Connection test failed: {test_response}")
                return False
        except Exception as e:
            print(f"❌ Connection test failed: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    # Create client instance with shorter timeout for testing
    client = GeminiClient(timeout=15)
    
    print("Example 1: Basic Response")
    prompt = "What is federated learning in one sentence?"
    response = client.get_response(prompt, timeout=10)
    print(f"Response: {response}")
    print("\n" + "=" * 30 + "\n")
       