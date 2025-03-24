# azure_openai_client.py
# This file defines the AzureOpenAIClient class, a utility for interacting with Azure OpenAI models 
# (gpt-4o, gpt-4o-mini, text-embedding-ada-002) in the text-to-SQL project. It initializes the API 
# client with model-specific configs, handles embeddings and token counting with tiktoken, tracks 
# costs using predefined pricing, and supports environment variable configuration via dotenv for 
# seamless integration across the pipeline.
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import tiktoken

load_dotenv()

class AzureOpenAIClient:
    # Pricing constants (per 1M tokens, as of March 2025)
    PRICING = {
        "gpt-4o": {
            "input": 2.50,  # $2.50 per 1M input tokens
            "output": 10.00  # $10.00 per 1M output tokens
        },
        "gpt-4o-mini": {
            "input": 0.15,  # $0.15 per 1M input tokens
            "cached_input": 0.075,  # $0.075 per 1M cached input tokens
            "output": 0.60  # $0.60 per 1M output tokens
        },
        "text-embedding-ada-002": {
            "input": 0.10  # $0.10 per 1M tokens (based on $0.0001 per 1K tokens)
        }
    }

    # Model-specific configurations
    MODEL_CONFIGS = {
        "gpt-4o": {
            "api_version": "2024-02-15-preview",
            "max_tokens": 128000  # Example max tokens for gpt-4o
        },
        "gpt-4o-mini": {
            "api_version": "2024-05-01-preview",
            "max_tokens": 128000  # Example max tokens for gpt-4o-mini
        },
        "text-embedding-ada-002": {
            "api_version": "2023-07-01-preview",
            "max_tokens": 8191  # Max tokens for text-embedding-ada-002
        }
    }

    def __init__(self, model_type="gpt-4o"):
        """
        Initialize the Azure OpenAI client with the specified model type.
        
        Args:
            model_type (str): Model name (e.g., 'gpt-4o', 'gpt-4o-mini', 'text-embedding-ada-002').
        """
        if model_type not in self.PRICING:
            raise ValueError(f"Unsupported model_type. Supported models: {list(self.PRICING.keys())}")
        
        self.model_type = model_type
        self.client = self._initialize_client()
        self.token_counter = 0  # Tracks total tokens processed
        self.encoding = tiktoken.get_encoding("cl100k_base") if model_type == "text-embedding-ada-002" else None

    def _initialize_client(self):
        """Initialize and return the Azure OpenAI client based on model type."""
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        if not all([api_key, endpoint]):
            raise ValueError("Azure OpenAI API key or endpoint not set. Check environment variables.")

        return AzureOpenAI(
            api_key=api_key,
            api_version=self.MODEL_CONFIGS[self.model_type]["api_version"],
            azure_endpoint=endpoint
        )

    def get_client(self):
        """Return the initialized Azure OpenAI client."""
        return self.client

    def get_model(self):
        """Return the model name."""
        return self.model_type

    def get_pricing(self):
        """Return the pricing for the selected model."""
        return self.PRICING[self.model_type]

    def count_tokens(self, text: str) -> int:
        """Return the approximate token count for a given text."""
        if self.model_type == "text-embedding-ada-002" and self.encoding:
            return len(self.encoding.encode(text))
        return len(text) // 4  # Rough estimate for other models (4 chars per token)

    def get_embedding(self, text: str) -> list:
        """Generate embedding for the given text using text-embedding-ada-002."""
        if self.model_type != "text-embedding-ada-002":
            raise ValueError("Embedding generation is only supported for 'text-embedding-ada-002'.")
        
        token_count = self.count_tokens(text)
        if token_count > self.MODEL_CONFIGS[self.model_type]["max_tokens"]:
            raise ValueError(f"Text exceeds max token limit of {self.MODEL_CONFIGS[self.model_type]['max_tokens']}.")

        response = self.client.embeddings.create(model=self.model_type, input=[text.replace("\n", " ")])
        embedding = response.data[0].embedding
        self.token_counter += token_count
        return embedding

    def calculate_cost(self) -> float:
        """Calculate the total cost based on tokens processed."""
        pricing = self.PRICING[self.model_type]
        if "input" in pricing:
            cost_per_token = pricing["input"] / 1_000_000  # Convert from per 1M to per token
            return self.token_counter * cost_per_token
        return 0.0

    def reset_token_counter(self):
        """Reset the token counter."""
        self.token_counter = 0