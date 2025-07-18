import openai
import json
import random
import re
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any
import logging

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DomainMetadataGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialize the metadata generator
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use (gpt-4 or gpt-3.5-turbo)
        """
        openai.api_key = api_key
        self.model = model
        self.chunk_size = 3000  # Characters per chunk for analysis
        self.samples_per_section = 3  # Number of samples from each section
    
    def smart_file_sampling(self, file_path: str) -> Dict[str, List[str]]:
        """
        Strategically sample content from beginning, middle, and end of file
        Handles both .txt and .json files
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Dictionary with samples from different sections
        """
        logger.info(f"Reading and sampling file: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.json':
            return self._sample_json_file(file_path)
        else:
            return self._sample_text_file(file_path)
    
    def _sample_text_file(self, file_path: str) -> Dict[str, List[str]]:
        """Handle .txt files"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Read file in chunks to handle very large files
            content_chunks = []
            while True:
                chunk = f.read(50000)  # Read 50KB at a time
                if not chunk:
                    break
                content_chunks.append(chunk)
        
        # Combine chunks
        full_content = ''.join(content_chunks)
        total_length = len(full_content)
        
        logger.info(f"Text file size: {total_length:,} characters")
        
        # Define section boundaries
        sections = {
            'beginning': (0, total_length // 4),
            'early_middle': (total_length // 4, total_length // 2),
            'late_middle': (total_length // 2, 3 * total_length // 4),
            'end': (3 * total_length // 4, total_length)
        }
        
        samples = {}
        
        for section_name, (start, end) in sections.items():
            section_content = full_content[start:end]
            section_samples = self._extract_meaningful_samples(
                section_content, 
                self.samples_per_section
            )
            samples[section_name] = section_samples
            logger.info(f"Extracted {len(section_samples)} samples from {section_name}")
        
        return samples
    
    def _sample_json_file(self, file_path: str) -> Dict[str, List[str]]:
        """Handle .json files by extracting text content and sampling strategically"""
        logger.info("Processing JSON file...")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                # Try to load as JSON
                data = json.load(f)
            except json.JSONDecodeError:
                # If not valid JSON, treat as text
                f.seek(0)
                return self._sample_text_file(file_path)
        
        # Extract all text content from JSON
        text_extracts = self._extract_text_from_json(data)
        
        if not text_extracts:
            logger.warning("No meaningful text found in JSON file")
            return {"beginning": [], "early_middle": [], "late_middle": [], "end": []}
        
        # Create sections from the extracted text
        total_extracts = len(text_extracts)
        logger.info(f"Extracted {total_extracts} text pieces from JSON")
        
        # Divide extracts into sections
        section_size = max(1, total_extracts // 4)
        
        sections = {
            'beginning': text_extracts[:section_size],
            'early_middle': text_extracts[section_size:2*section_size],
            'late_middle': text_extracts[2*section_size:3*section_size], 
            'end': text_extracts[3*section_size:]
        }
        
        samples = {}
        for section_name, section_extracts in sections.items():
            # Sample from each section
            if len(section_extracts) > self.samples_per_section:
                section_samples = random.sample(section_extracts, self.samples_per_section)
            else:
                section_samples = section_extracts
            
            # Filter samples by length
            section_samples = [
                sample for sample in section_samples 
                if len(sample.strip()) >= 100
            ]
            
            samples[section_name] = section_samples[:self.samples_per_section]
            logger.info(f"Selected {len(samples[section_name])} samples from {section_name}")
        
        return samples
    
    def _extract_text_from_json(self, data: Any, max_extracts: int = 1000) -> List[str]:
        """
        Recursively extract meaningful text content from JSON structure
        
        Args:
            data: JSON data (dict, list, or primitive)
            max_extracts: Maximum number of text extracts to collect
            
        Returns:
            List of meaningful text strings
        """
        text_extracts = []
        
        def extract_recursive(obj, path=""):
            if len(text_extracts) >= max_extracts:
                return
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    
                    # If the value is a string and looks meaningful
                    if isinstance(value, str) and self._is_meaningful_text(value):
                        # Include key context for better understanding
                        context_text = f"[{key}]: {value}"
                        text_extracts.append(context_text)
                    else:
                        extract_recursive(value, new_path)
                        
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_path = f"{path}[{i}]" if path else f"[{i}]"
                    
                    if isinstance(item, str) and self._is_meaningful_text(item):
                        text_extracts.append(item)
                    else:
                        extract_recursive(item, new_path)
                        
            elif isinstance(obj, str) and self._is_meaningful_text(obj):
                text_extracts.append(obj)
        
        extract_recursive(data)
        return text_extracts
    
    def _is_meaningful_text(self, text: str) -> bool:
        """
        Check if a string contains meaningful text content
        
        Args:
            text: String to check
            
        Returns:
            True if text appears meaningful
        """
        if not isinstance(text, str):
            return False
            
        text = text.strip()
        
        # Filter criteria
        if len(text) < 10:  # Too short
            return False
        if len(text) > 5000:  # Too long for a single extract
            return False
        if text.isdigit():  # Pure numbers
            return False
        if len(text.split()) < 3:  # Too few words
            return False
        if re.match(r'^[\d\-\s\.\,\:\;]+$', text):  # Only dates/numbers/punctuation
            return False
        if text.startswith('http'):  # URLs
            return False
        if len(set(text.replace(' ', ''))) < 5:  # Too repetitive
            return False
            
        return True
    
    def _extract_meaningful_samples(self, content: str, num_samples: int) -> List[str]:
        """
        Extract meaningful text chunks from content section
        
        Args:
            content: Text content to sample from
            num_samples: Number of samples to extract
            
        Returns:
            List of meaningful text samples
        """
        # Split by paragraphs/sections (try different delimiters)
        delimiters = ['\n\n\n', '\n\n', '\n---\n', '\n##', '\n#']
        
        chunks = [content]
        for delimiter in delimiters:
            if len(chunks[0]) > self.chunk_size * 2:  # Only split if chunks are too large
                new_chunks = []
                for chunk in chunks:
                    new_chunks.extend(chunk.split(delimiter))
                chunks = [c.strip() for c in new_chunks if len(c.strip()) > 100]
        
        # Filter out very short or very long chunks
        filtered_chunks = [
            chunk for chunk in chunks 
            if 200 <= len(chunk) <= self.chunk_size * 2
        ]
        
        # If not enough good chunks, use sliding window
        if len(filtered_chunks) < num_samples:
            step_size = len(content) // (num_samples + 1)
            filtered_chunks = []
            for i in range(num_samples):
                start = i * step_size
                end = start + self.chunk_size
                chunk = content[start:end]
                if len(chunk.strip()) > 200:
                    filtered_chunks.append(chunk.strip())
        
        # Randomly sample if we have more than needed
        if len(filtered_chunks) > num_samples:
            filtered_chunks = random.sample(filtered_chunks, num_samples)
        
        return filtered_chunks[:num_samples]
    
    def analyze_sample_batch(self, samples: List[str], section_name: str) -> str:
        """
        Analyze a batch of samples using OpenAI
        
        Args:
            samples: List of text samples
            section_name: Name of the section being analyzed
            
        Returns:
            Analysis results as string
        """
        combined_samples = '\n\n--- SAMPLE SEPARATOR ---\n\n'.join(samples)
        
        prompt = f"""
        Analyze these {len(samples)} content samples from the {section_name} section of a document:

        SAMPLES:
        {combined_samples[:4000]}  # Limit to stay within token limits
        
        Based on these samples, identify:
        1. Main topics and themes
        2. Technical terminology and keywords
        3. Content type (documentation, data, reports, code, etc.)
        4. Domain/subject area
        5. Types of questions this content could answer
        
        Respond in this format:
        TOPICS: [comma-separated main topics]
        KEYWORDS: [comma-separated technical terms and important keywords]
        CONTENT_TYPE: [type of content]
        DOMAIN: [subject domain]
        QUESTION_TYPES: [types of questions this could answer]
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",  # Use cheaper model for individual analyses
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error analyzing {section_name}: {e}")
            return f"Error analyzing {section_name}: {str(e)}"
    
    def synthesize_final_metadata(self, section_analyses: Dict[str, str], domain_name: str) -> Dict:
        """
        Synthesize all section analyses into final metadata
        
        Args:
            section_analyses: Dictionary of section analysis results
            domain_name: Name of the domain
            
        Returns:
            Final metadata dictionary
        """
        combined_analysis = "\n\n".join([
            f"=== {section.upper()} SECTION ===\n{analysis}"
            for section, analysis in section_analyses.items()
        ])
        
        prompt = f"""
        Based on these comprehensive analyses from different sections of a document, create final domain metadata for a RAG routing system:

        SECTION ANALYSES:
        {combined_analysis}
        
        Create comprehensive metadata in valid JSON format:
        {{
            "description": "Clear, concise 1-2 sentence description of what this dataset contains",
            "keywords": ["15-25 most important and diverse keywords for query matching"],
            "example_queries": ["8-12 diverse example questions users might ask"],
            "main_topics": ["6-10 main topic areas covered"],
            "content_types": ["types of content: reports, documentation, data, etc."],
            "domain_focus": "primary domain/subject area",
            "query_patterns": ["common patterns in how users might phrase questions"]
        }}
        
        Ensure the JSON is valid and includes diverse keywords that would help route different types of queries correctly.
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Extract JSON from response
            response_text = response.choices[0].message.content
            
            # Find JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # Fallback parsing
                return self._parse_fallback_metadata(response_text, domain_name)
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return self._parse_fallback_metadata(response.choices[0].message.content, domain_name)
        except Exception as e:
            logger.error(f"Error in synthesis: {e}")
            return self._create_fallback_metadata(domain_name)
    
    def _parse_fallback_metadata(self, response_text: str, domain_name: str) -> Dict:
        """Fallback parser if JSON parsing fails"""
        logger.warning("Using fallback metadata parsing")
        
        # Extract information using regex patterns
        description_match = re.search(r'"description":\s*"([^"]*)"', response_text)
        keywords_match = re.search(r'"keywords":\s*\[(.*?)\]', response_text, re.DOTALL)
        
        metadata = {
            "description": description_match.group(1) if description_match else f"Content from {domain_name} dataset",
            "keywords": [],
            "example_queries": [f"What information is in {domain_name}?"],
            "main_topics": [domain_name],
            "content_types": ["mixed content"],
            "domain_focus": domain_name,
            "query_patterns": ["general queries"]
        }
        
        if keywords_match:
            keywords_str = keywords_match.group(1)
            keywords = re.findall(r'"([^"]*)"', keywords_str)
            metadata["keywords"] = keywords[:20]
        
        return metadata
    
    def _create_fallback_metadata(self, domain_name: str) -> Dict:
        """Create basic fallback metadata"""
        return {
            "description": f"Content from {domain_name} dataset",
            "keywords": [domain_name, "information", "data"],
            "example_queries": [f"What is in {domain_name}?"],
            "main_topics": [domain_name],
            "content_types": ["mixed content"],
            "domain_focus": domain_name,
            "query_patterns": ["general queries"]
        }
    
    def generate_metadata_for_file(self, file_path: str, domain_name: str) -> Dict:
        """
        Complete pipeline to generate metadata for a single file
        
        Args:
            file_path: Path to the file
            domain_name: Name of the domain
            
        Returns:
            Complete metadata dictionary
        """
        logger.info(f"Starting metadata generation for {domain_name}")
        
        # Step 1: Smart sampling
        samples = self.smart_file_sampling(file_path)
        
        # Step 2: Analyze each section
        section_analyses = {}
        for section_name, section_samples in samples.items():
            if section_samples:  # Only analyze if we have samples
                logger.info(f"Analyzing {section_name} section...")
                analysis = self.analyze_sample_batch(section_samples, section_name)
                section_analyses[section_name] = analysis
        
        # Step 3: Synthesize final metadata
        logger.info("Synthesizing final metadata...")
        final_metadata = self.synthesize_final_metadata(section_analyses, domain_name)
        
        logger.info(f"Metadata generation complete for {domain_name}")
        return final_metadata
    
    def generate_routing_metadata(self, file_configs: List[Tuple[str, str]]) -> Dict:
        """
        Generate complete routing metadata for multiple files
        
        Args:
            file_configs: List of (file_path, domain_name) tuples
            
        Returns:
            Complete domain metadata dictionary
        """
        domain_metadata = {}
        
        for file_path, domain_name in file_configs:
            try:
                metadata = self.generate_metadata_for_file(file_path, domain_name)
                domain_metadata[domain_name] = metadata
                logger.info(f"✅ Successfully generated metadata for {domain_name}")
            except Exception as e:
                logger.error(f"❌ Failed to generate metadata for {domain_name}: {e}")
                domain_metadata[domain_name] = self._create_fallback_metadata(domain_name)
        
        return domain_metadata

def main():
    """
    Example usage of the DomainMetadataGenerator
    """
    # Configuration
    API_KEY = os.getenv("OPENAI_API_KEY")  # Get from environment variable
    
    if not API_KEY:
        print("❌ Error: OPENAI_API_KEY environment variable not found!")
        print("Please set it in your .env file or environment variables")
        return
    
    # File configurations: (file_path, domain_name)
    file_configs = [
        ("/Users/krishnayadav/Documents/test_projects/polkassembly-ai-v2/polkassembly-ai/data/joined_data/static/combined.txt", "static_documentation"),
        ("/Users/krishnayadav/Documents/test_projects/polkassembly-ai-v2/polkassembly-ai/data/joined_data/dynamic/combined.json", "dynamic_documentation")
    ]
    
    # Validate file paths
    for file_path, domain_name in file_configs:
        if not os.path.exists(file_path):
            print(f"❌ Warning: File not found: {file_path}")
            print(f"   Please check the path for {domain_name}")
    
    # Initialize generator
    generator = DomainMetadataGenerator(
        api_key=API_KEY,
        model="gpt-4"  # or "gpt-3.5-turbo" for cheaper option
    )
    
    # Generate metadata
    print("🚀 Starting domain metadata generation...")
    domain_metadata = generator.generate_routing_metadata(file_configs)
    
    # Save results
    output_file = "domain_metadata.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(domain_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Domain metadata saved to {output_file}")
    
    # Display summary
    print("\n📊 Generated Metadata Summary:")
    for domain, metadata in domain_metadata.items():
        print(f"\n🏷️  Domain: {domain}")
        print(f"   Description: {metadata.get('description', 'N/A')}")
        print(f"   Keywords: {len(metadata.get('keywords', []))} keywords")
        print(f"   Example Queries: {len(metadata.get('example_queries', []))} queries")
        print(f"   Main Topics: {', '.join(metadata.get('main_topics', []))}")

# Example for testing individual file types
def test_file_type_handling():
    """Test the file type handling separately"""
    API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not API_KEY:
        print("❌ Error: OPENAI_API_KEY environment variable not found!")
        return
    
    generator = DomainMetadataGenerator(api_key=API_KEY)
    
    # Test JSON file
    json_file = "/Users/krishnayadav/Documents/test_projects/polkassembly-ai-v2/polkassembly-ai/data/joined_data/dynamic/combined.json"
    if os.path.exists(json_file):
        print("Testing JSON file handling...")
        json_samples = generator.smart_file_sampling(json_file)
        print(f"JSON samples extracted: {sum(len(samples) for samples in json_samples.values())}")
    
    # Test text file  
    txt_file = "/Users/krishnayadav/Documents/test_projects/polkassembly-ai-v2/polkassembly-ai/data/joined_data/static/combined.txt"
    if os.path.exists(txt_file):
        print("Testing text file handling...")
        txt_samples = generator.smart_file_sampling(txt_file)
        print(f"Text samples extracted: {sum(len(samples) for samples in txt_samples.values())}")

if __name__ == "__main__":
    main()
    # Uncomment to test file handling separately:
    # test_file_type_handling()