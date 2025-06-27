import tiktoken
import logging
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextChunker:
    """Split documents into smaller chunks for embedding generation"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, encoding_name: str = "cl100k_base"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            return len(text) // 4  # Rough approximation
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a single document into chunks
        
        Args:
            document: Dictionary with 'content' and 'metadata' keys
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        content = document['content']
        metadata = document['metadata']
        
        if not content or not content.strip():
            return []
        
        # Split the text into chunks
        chunks = self.text_splitter.split_text(content)
        
        chunk_documents = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Only add non-empty chunks
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_index'] = i
                chunk_metadata['total_chunks'] = len(chunks)
                chunk_metadata['chunk_size'] = len(chunk)
                chunk_metadata['chunk_tokens'] = self._count_tokens(chunk)
                
                chunk_documents.append({
                    'content': chunk.strip(),
                    'metadata': chunk_metadata
                })
        
        return chunk_documents
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split multiple documents into chunks
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc_idx, document in enumerate(documents):
            try:
                chunks = self.chunk_document(document)
                
                # Add document index to each chunk
                for chunk in chunks:
                    chunk['metadata']['document_index'] = doc_idx
                
                all_chunks.extend(chunks)
                
                if doc_idx % 10 == 0:
                    logger.info(f"Processed {doc_idx + 1}/{len(documents)} documents")
                    
            except Exception as e:
                logger.error(f"Error chunking document {doc_idx}: {e}")
                continue
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the chunks"""
        if not chunks:
            return {}
        
        chunk_sizes = [chunk['metadata']['chunk_size'] for chunk in chunks]
        chunk_tokens = [chunk['metadata']['chunk_tokens'] for chunk in chunks]
        
        stats = {
            'total_chunks': len(chunks),
            'average_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'average_tokens': sum(chunk_tokens) / len(chunk_tokens),
            'min_tokens': min(chunk_tokens),
            'max_tokens': max(chunk_tokens),
            'total_tokens': sum(chunk_tokens)
        }
        
        # Count chunks by source
        source_counts = {}
        for chunk in chunks:
            source = chunk['metadata']['source']
            source_counts[source] = source_counts.get(source, 0) + 1
        
        stats['chunks_by_source'] = source_counts
        
        return stats
    
    def filter_chunks_by_size(self, chunks: List[Dict[str, Any]], 
                             min_tokens: int = 50, max_tokens: int = None) -> List[Dict[str, Any]]:
        """Filter chunks by token size"""
        if max_tokens is None:
            max_tokens = self.chunk_size
        
        filtered_chunks = []
        for chunk in chunks:
            token_count = chunk['metadata']['chunk_tokens']
            if min_tokens <= token_count <= max_tokens:
                filtered_chunks.append(chunk)
        
        logger.info(f"Filtered {len(chunks)} chunks to {len(filtered_chunks)} chunks "
                   f"(min_tokens={min_tokens}, max_tokens={max_tokens})")
        
        return filtered_chunks 