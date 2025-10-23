"""
Prompt-based Graph Encoder

Module for generating graphs from instructions based on YAML prompts
"""

import json
import yaml
import sys
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import concurrent.futures
import threading
from pathlib import Path

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

# Add project path
sys.path.append("/HabitatLLM")

try:
    from src.models.base import ModelFactory
    MODELFACTORY_AVAILABLE = True
except ImportError:
    MODELFACTORY_AVAILABLE = False
    ModelFactory = None

from graph_schema import (
    NodeType, EdgeType, EdgeRelation, KnowledgeSubtype, ObjectGranularity,
    GraphNode, UserNode, KnowledgeNode, PatternNode, ObjectNode, LocationNode,
    GraphEdge, SemanticGraph
)

class PromptBasedGraphEncoder:
    """
    Class for converting instructions to semantic graphs based on YAML prompts
    """
    
    def __init__(self, prompt_file: str = None, api_key: str = None, model: str = "gpt-4o", 
                 use_openrouter: bool = False, reasoning: bool = False):
        """
        Args:
            prompt_file: Path to YAML prompt file
            api_key: OpenAI API key or OpenRouter API key
            model: LLM model name to use
            use_openrouter: Whether to use OpenRouter API
            reasoning: Whether to use reasoning mode
        """
        self.model = model
        self.use_openrouter = use_openrouter
        self.reasoning = reasoning
        self.api_key = api_key
        self.prompt_file = prompt_file or "/HabitatLLM/rebuttal/structurize/prompts/prompt.yaml"
        
        # ModelFactory or OpenAI setup
        if use_openrouter and MODELFACTORY_AVAILABLE:
            # Create config object for OpenRouter usage
            class Config:
                def __init__(self):
                    self.api_key = api_key
                    self.use_openrouter = True
                    self.reasoning = reasoning
                
                def get(self, key, default=None):
                    return getattr(self, key, default)
            
            self.config = Config()
            try:
                self.model_factory = ModelFactory(model_name=model, config=self.config)
                self.llm_model = self.model_factory.get_model()
            except Exception as e:
                print(f"Failed to initialize OpenRouter model: {e}")
                self.llm_model = None
        elif OPENAI_AVAILABLE and api_key and not use_openrouter:
            # OpenAI setup
            openai.api_key = api_key
            self.llm_model = None
        else:
            if use_openrouter and not MODELFACTORY_AVAILABLE:
                print("Warning: ModelFactory not available. Cannot use OpenRouter.")
            elif not OPENAI_AVAILABLE:
                print("Warning: OpenAI not available. Using mock mode.")
            self.llm_model = None
        
        # Load YAML prompts
        self.prompt_templates = self._load_prompt_templates()
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates from YAML file"""
        try:
            with open(self.prompt_file, 'r', encoding='utf-8') as f:
                prompt_data = yaml.safe_load(f)
            
            return {
                'knowledge_classifier': prompt_data.get('knowledge_classifier', ''),
                'graph_encoder': prompt_data.get('graph_encoder', '')
            }
        except Exception as e:
            print(f"Error loading prompt templates: {e}")
            return {
                'knowledge_classifier': 'Classify the knowledge type as object_semantics or user_pattern.',
                'graph_encoder': 'You are a graph encoder. Convert instructions to JSON graph format.'
            }
    
    def classify_knowledge_type(self, instruction: str) -> str:
        """
        Analyze instruction and classify knowledge_type
        
        Args:
            instruction: Instruction to classify
            
        Returns:
            knowledge_type: "object_semantics" or "user_pattern"
        """
        prompt = self.prompt_templates['knowledge_classifier'].format(instruction=instruction)
        
        try:
            response = self._call_llm(prompt)
            
            # Extract knowledge_type from response
            if "object_semantics" in response.lower():
                return "object_semantics"
            elif "user_pattern" in response.lower():
                return "user_pattern"
            else:
                # Return user_pattern as default (recommended in prompt)
                print(f"Warning: Could not determine knowledge_type from response: {response[:201]}")
                return "user_pattern"
                
        except Exception as e:
            print(f"Error in classify_knowledge_type: {e}")
            return "user_pattern"  # Default value
    
    def encode_instruction(self, instruction: str, user_name: str = "user_A", episode_id: str = "", answer_knowledge_type: str = None) -> SemanticGraph:
        """
        Convert single instruction to semantic graph (2-step processing)
        
        Args:
            instruction: User instruction
            user_name: User name
            episode_id: Episode ID
            answer_knowledge_type: Correct knowledge_type (use if available, otherwise classify)
            
        Returns:
            Generated SemanticGraph object
        """
        # Step 1: Knowledge Type classification (always classify with LLM)
        knowledge_type = self.classify_knowledge_type(instruction)
        print(f"Classified knowledge_type: {knowledge_type}")
        
        # Compare with ground truth (for evaluation)
        if answer_knowledge_type:
            is_correct = (knowledge_type == answer_knowledge_type)
            print(f"Ground truth: {answer_knowledge_type}, Prediction: {knowledge_type}, Correct: {is_correct}")
        
        # Step 2: Graph generation
        full_prompt = self.prompt_templates['graph_encoder'].format(
            instruction=instruction,
            knowledge_type=knowledge_type
        )
        
        # Call LLM
        response = self._call_llm(full_prompt)
        # Parse JSON response
        graph_data = self._parse_llm_response(response)
        
        # Add knowledge_type if missing
        if 'knowledge_type' not in graph_data:
            graph_data['knowledge_type'] = knowledge_type
            
        # Convert to SemanticGraph object
        semantic_graph = self._build_semantic_graph(graph_data, user_name, episode_id, answer_knowledge_type)
        return semantic_graph
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM API"""
        if self.use_openrouter and self.llm_model:
            # Use OpenRouter
            try:
                # Adjust system message by model
                if "llama" in self.model.lower():
                    system_msg = "You are an expert at extracting semantic information from text. Always return valid JSON with the exact format requested. Do not return empty arrays unless there is truly no semantic content."
                elif "qwen" in self.model.lower():
                    system_msg = "You are a helpful assistant that extracts semantic information from instructions. Return only valid JSON in the exact format requested."
                else:
                    system_msg = "You are a helpful assistant that extracts semantic information from instructions."
                    
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ]
                
                responses = self.llm_model.generate_response(
                    messages=messages,
                    temperature=0.0,
                    max_tokens=2000
                )
                
                return responses[0].strip()
                
            except Exception as e:
                raise Exception(f"OpenRouter API call failed: {str(e)}")
        
        elif not self.use_openrouter and OPENAI_AVAILABLE:
            # Use OpenAI
            try:
                # OpenAI 1.0+ version compatibility
                from openai import OpenAI
                client = OpenAI()
                
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=2000
                )
                return response.choices[0].message.content.strip()
            except ImportError:
                # Legacy OpenAI version
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=2000
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                raise Exception(f"OpenAI API call failed: {str(e)}")
        
        else:
            # When API is not available
            if self.use_openrouter:
                raise Exception("OpenRouter model not available. Check ModelFactory initialization.")
            else:
                raise Exception("OpenAI module not available. Install with: pip install openai")
    
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to JSON"""
        try:
            # Extract only JSON part
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.rfind("```")
                json_str = response[json_start:json_end].strip()
            elif response.strip().startswith("{"):
                json_str = response.strip()
            else:
                # Return default structure if not in JSON format
                raise ValueError("Response is not in JSON format")
            
            parsed_data = json.loads(json_str)
            print(f"Successfully parsed JSON: {json_str[:201]}...")
            return parsed_data
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse LLM response as JSON: {str(e)}")
            print(f"Response preview: {response[:200]}...")
            # Explicitly indicate JSON parsing failure
            raise ValueError(f"JSON parsing failed: {str(e)}")
    
    def _build_semantic_graph(self, graph_data: Dict[str, Any], user_name: str, episode_id: str = "", answer_knowledge_type: str = None) -> SemanticGraph:
        """Convert parsed JSON data to SemanticGraph object"""
        graph = SemanticGraph()
        
        # Create nodes
        for node_data in graph_data.get("nodes", []):
            node = self._create_node_from_data(node_data, user_name, episode_id, answer_knowledge_type)
            if node:
                graph.add_node(node)
        
        # Create edges
        for edge_data in graph_data.get("edges", []):
            edge = self._create_edge_from_data(edge_data)
            if edge:
                graph.add_edge(edge)
        
        # Set metadata
        if answer_knowledge_type:
            graph.answer_knowledge_type = answer_knowledge_type
        
        return graph
    
    def _create_node_from_data(self, node_data: Dict[str, Any], user_name: str, episode_id: str = "", answer_knowledge_type: str = None) -> Optional[GraphNode]:
        """Convert JSON node data to GraphNode object"""
        try:
            node_type = NodeType(node_data["type"])
            node_id = node_data["id"]
            
            if node_type == NodeType.USER:
                name = node_data.get("name", user_name)
                return UserNode(id=node_id, type=node_type, name=name)
            
            elif node_type == NodeType.KNOWLEDGE:
                subtype = KnowledgeSubtype(node_data["subtype"])
                alias = node_data.get("alias", "unknown knowledge")
                description = node_data.get("description", "")
                return KnowledgeNode(id=node_id, type=node_type, subtype=subtype, alias=alias, description=description)
            
            elif node_type == NodeType.PATTERN:
                name = node_data.get("name", "unknown pattern")
                args = node_data.get("args", [])
                return PatternNode(id=node_id, type=node_type, name=name, args=args)
            
            elif node_type == NodeType.OBJECT:
                name = node_data.get("name", "unknown object")
                granularity_str = node_data.get("granularity", "type")
                granularity = ObjectGranularity(granularity_str)
                attributes = node_data.get("attributes", [])
                return ObjectNode(id=node_id, type=node_type, name=name, granularity=granularity, attributes=attributes)
            
            elif node_type == NodeType.LOCATION:
                name = node_data.get("name", "unknown location")
                expression = node_data.get("expression", "unknown expression")
                return LocationNode(id=node_id, type=node_type, name=name, expression=expression)
            
        except (KeyError, ValueError) as e:
            print(f"Warning: Failed to create node from data {node_data}: {e}")
            return None
    
    def _create_edge_from_data(self, edge_data: Dict[str, Any]) -> Optional[GraphEdge]:
        """Convert JSON edge data to GraphEdge object"""
        try:
            source_id = edge_data.get("source") or edge_data.get("source_id")
            target_id = edge_data.get("target") or edge_data.get("target_id")
            relation_str = edge_data.get("relation")
            edge_type_str = edge_data.get("type", "Hierarchical")  # Default is Hierarchical
            object_ref = edge_data.get("object")  # optional object reference
            
            if not source_id or not target_id or not relation_str:
                raise ValueError("Missing required edge fields")
            
            relation = EdgeRelation(relation_str)
            edge_type = EdgeType(edge_type_str)
            
            return GraphEdge(
                source=source_id,
                target=target_id,
                type=edge_type,
                relation=relation,
                object=object_ref
            )
        except (KeyError, ValueError) as e:
            print(f"Warning: Failed to create edge from data {edge_data}: {e}")
            return None


class PromptBasedGraphManager:
    """
    Class for managing graphs using PromptBasedGraphEncoder
    """
    
    def __init__(self, output_dir: str = "/HabitatLLM/rebuttal/structurize/test_output", model_name: str = None, use_openrouter: bool = False):
        """
        Args:
            output_dir: Directory to save output files
            model_name: Model name (for folder name generation)
            use_openrouter: Whether to use OpenRouter
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model-specific folder
        if model_name:
            self.model_folder = self._generate_model_folder_name(model_name, use_openrouter)
            self.model_output_dir = self.output_dir / self.model_folder
            self.model_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.model_folder = None
            self.model_output_dir = self.output_dir
            
        self.graphs = []  # Store graphs in memory
        self.failed_episodes = []  # Accumulate failed episodes
        self.lock = threading.Lock()  # Lock for thread safety
    
    def _generate_model_folder_name(self, model_name: str, use_openrouter: bool = False) -> str:
        """Generate folder name from model name"""
        if use_openrouter:
            # OpenRouter model name conversion (e.g., meta-llama/llama-3.1-8b-instruct -> openrouter_meta_llama_llama_3_1_8b_instruct)
            clean_name = model_name.replace('/', '_').replace('-', '_').replace('.', '_')
            return f"openrouter_{clean_name}"
        else:
            # OpenAI model name conversion (e.g., gpt-4o -> openai_gpt_4o)
            clean_name = model_name.replace('-', '_').replace('.', '_')
            return f"openai_{clean_name}"
    
    def process_instruction(self, instruction: str, encoder: PromptBasedGraphEncoder, 
                          output_prefix: str = "test", episode_id: str = "", scene_id: str = "", answer_knowledge_type: str = None) -> Dict[str, Any]:
        """
        Process single instruction to generate and save graph
        
        Args:
            instruction: Instruction to process
            encoder: PromptBasedGraphEncoder instance
            output_prefix: Output filename prefix
            episode_id: Episode ID
            scene_id: Scene ID
            answer_knowledge_type: Correct knowledge_type (for evaluation)
            
        Returns:
            Processing result information
        """
        try:
            # Generate graph
            graph = encoder.encode_instruction(instruction, episode_id=episode_id, answer_knowledge_type=answer_knowledge_type)
            
            # Set metadata
            graph.episode_id = episode_id
            graph.scene_id = scene_id
            graph.instruction = instruction
            graph.answer_knowledge_type = answer_knowledge_type
            # Store graph in memory (thread-safe)
            with self.lock:
                self.graphs.append(graph)
            
            # Prepare embedding data (only when needed)
            # embedding_data = self._prepare_embedding_data(graph)
            # embedding_file = self.output_dir / f"{output_prefix}_embedding_data.json"
            # self._save_embedding_data(embedding_data, embedding_file)
            
            return {
                "success": True,
                "instruction": instruction,
                "episode_id": episode_id,
                "scene_id": scene_id,
                "answer_knowledge_type": answer_knowledge_type,
                "nodes_count": len(graph.nodes),
                "edges_count": len(graph.edges),
                "knowledge_type": self._extract_knowledge_type(graph)
            }
        
        except Exception as e:
            return {
                "success": False,
                "instruction": instruction,
                "error": str(e)
            }
    
    def _save_graph(self, graph: SemanticGraph, file_path: Path):
        """Save graph to JSON file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(graph.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Graph saved to {file_path}")
    
    def _save_embedding_data(self, embedding_data: Dict[str, Any], file_path: Path):
        """Save embedding data to JSON file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(embedding_data, f, indent=2, ensure_ascii=False)
        print(f"Embedding data saved to {file_path}")
    
    def _prepare_embedding_data(self, graph: SemanticGraph) -> Dict[str, Any]:
        """Prepare embedding data from graph"""
        embedding_data = {
            "node_texts": {},
            "edge_descriptions": [],
            "graph_summary": {}
        }
        
        # Generate text for each node
        for node_id, node in graph.nodes.items():
            if node.type == NodeType.KNOWLEDGE:
                text = f"Knowledge: {node.alias}"
                if hasattr(node, 'description') and node.description:
                    text += f" - {node.description}"
            elif node.type == NodeType.OBJECT:
                text = f"Object: {node.name}"
                if node.attributes:
                    text += f" ({', '.join(node.attributes)})"
            elif node.type == NodeType.LOCATION:
                text = f"Location: {node.name} - {node.expression}"
            elif node.type == NodeType.PATTERN:
                text = f"Pattern: {node.name}"
                if node.args:
                    text += f" with args {', '.join(node.args)}"
            elif node.type == NodeType.USER:
                text = f"User: {node.name}"
            else:
                text = f"{node.type.value}: {node.id}"
            
            embedding_data["node_texts"][node_id] = text
        
        # Generate edge descriptions
        for edge in graph.edges:
            source_text = embedding_data["node_texts"].get(edge.source_id, edge.source_id)
            target_text = embedding_data["node_texts"].get(edge.target_id, edge.target_id)
            edge_desc = f"{source_text} {edge.relation.value} {target_text}"
            embedding_data["edge_descriptions"].append(edge_desc)
        
        # Graph summary
        embedding_data["graph_summary"] = {
            "total_nodes": len(graph.nodes),
            "total_edges": len(graph.edges),
            "node_types": {nt.value: len([n for n in graph.nodes.values() if n.type == nt]) 
                          for nt in NodeType},
            "edge_types": {et.value: len([e for e in graph.edges if e.relation == et]) 
                          for et in EdgeType}
        }
        
        return embedding_data
    
    def _extract_knowledge_type(self, graph: SemanticGraph) -> Optional[str]:
        """Extract knowledge type from graph"""
        for node in graph.nodes.values():
            if node.type == NodeType.KNOWLEDGE:
                return node.subtype.value
        return None
    
    def save_graphs_by_scene(self, output_dir: str = None):
        """
        Group and save graphs by Scene ID (in model-specific folder)
        """
        if output_dir is None:
            output_dir = str(self.model_output_dir)  # Use model-specific folder
            
        scene_graphs = {}
        
        # Classify graphs in memory by scene_id
        for graph in self.graphs:
            scene_id = graph.scene_id if graph.scene_id else 'unknown'
            if scene_id not in scene_graphs:
                scene_graphs[scene_id] = []
            
            scene_graphs[scene_id].append(graph.to_dict())
        
        # Save by Scene ID (merge with existing files)
        scene_output_dir = Path(output_dir) / "scenes"
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        
        for scene_id, new_graphs in scene_graphs.items():
            scene_file = scene_output_dir / f"scene_{scene_id}_graphs.json"
            
            # Load and merge if existing file exists
            existing_graphs = []
            if scene_file.exists():
                try:
                    with open(scene_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        existing_graphs = existing_data.get('graphs', [])
                    print(f"Found existing {len(existing_graphs)} graphs for scene {scene_id}")
                except Exception as e:
                    print(f"Error reading existing file {scene_file}: {e}")
                    existing_graphs = []
            
            # Merge new graphs with existing graphs
            all_graphs = existing_graphs + new_graphs
            
            scene_data = {
                "scene_id": scene_id,
                "total_graphs": len(all_graphs),
                "graphs": all_graphs
            }
            
            with open(scene_file, 'w', encoding='utf-8') as f:
                json.dump(scene_data, f, indent=2, ensure_ascii=False)
            
            print(f"Saved {len(all_graphs)} total graphs for scene {scene_id} (new: {len(new_graphs)}, existing: {len(existing_graphs)})")
        
        print(f"Total scenes processed: {len(scene_graphs)}")
        if self.model_folder:
            print(f"All graphs saved in model folder: {self.model_folder}")
        
        # Clean up memory after saving
        self.graphs = []
    
    def process_instructions_parallel(self, episodes: List[Dict], encoder: PromptBasedGraphEncoder, max_workers: int = 201) -> List[Dict[str, Any]]:
        """
        Process multiple instructions in parallel
        
        Args:
            episodes: List of episodes to process
            encoder: PromptBasedGraphEncoder instance
            max_workers: Maximum number of workers
            
        Returns:
            List of processing results
        """
        results = []
        
        def process_single_episode(episode_data):
            episode, episode_num = episode_data
            instruction = episode['instruction']
            episode_id = str(episode.get('episode_id', episode_num))
            scene_id = str(episode.get('scene_id', 'unknown'))
            
            # Extract episode_type from metadata (ground truth value)
            answer_knowledge_type = None
            metadata = episode.get('metadata', {})
            if metadata and 'episode_type' in metadata:
                answer_knowledge_type = metadata['episode_type']
            
            print(f"Processing Episode {episode_num} (Thread: {threading.current_thread().name})")
            
            result = self.process_instruction(
                instruction, encoder, f"test_{episode_num}", 
                episode_id=episode_id, scene_id=scene_id, answer_knowledge_type=answer_knowledge_type
            )
            result['episode_num'] = episode_num
            return result
        
        # Prepare episode data with numbers together
        episode_data_list = [(episode, i+1) for i, episode in enumerate(episodes)]
        
        # Parallel processing with ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_episode = {
                executor.submit(process_single_episode, episode_data): episode_data 
                for episode_data in episode_data_list
            }
            
        # Collect results
        failed_episodes = []  # Track failed episodes
        
        for future in concurrent.futures.as_completed(future_to_episode):
            try:
                result = future.result()
                results.append(result)
                
                if result["success"]:
                    print(f"✓ Episode {result['episode_num']}: {result['nodes_count']} nodes, {result['edges_count']} edges")
                else:
                    print(f"✗ Episode {result['episode_num']}: {result['error']}")
                    # Save failed episode information
                    episode_data = future_to_episode[future]
                    failed_episodes.append({
                        'episode_data': episode_data[0],
                        'episode_num': result['episode_num'],
                        'error': result['error']
                    })
                    
            except Exception as e:
                episode_data = future_to_episode[future]
                print(f"✗ Episode {episode_data[1]} failed with exception: {e}")
                # Add episodes that raised exceptions
                failed_episodes.append({
                    'episode_data': episode_data[0],
                    'episode_num': episode_data[1],
                    'error': str(e)
                })
        
        # Accumulate failed episodes (thread-safe)
        if failed_episodes:
            with self.lock:
                self.failed_episodes.extend(failed_episodes)
            print(f"\nAccumulated {len(failed_episodes)} failed episodes. (Total: {len(self.failed_episodes)})")
        
        return results
    
    def _save_failed_episodes(self, failed_episodes: List[Dict[str, Any]]):
        """Save failed episodes to JSON file"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/HabitatLLM/rebuttal/structurize/failed_episodes_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(failed_episodes, f, ensure_ascii=False, indent=2)
            print(f"Failed episode information saved: {filename}")
        except Exception as e:
            print(f"Error saving failed episodes: {e}")
    
    def retry_failed_episodes(self, failed_episodes_file: str, encoder, max_workers: int = 50) -> Dict[str, Any]:
        """Reprocess only failed episodes"""
        print(f"=== Starting failed episode reprocessing: {failed_episodes_file} ===")
        
        try:
            with open(failed_episodes_file, 'r', encoding='utf-8') as f:
                failed_episodes = json.load(f)
        except Exception as e:
            print(f"Error reading failed episode file: {e}")
            return {'success': False, 'error': str(e)}
        
        print(f"Number of episodes to reprocess: {len(failed_episodes)}")
        
        # Extract only episode data
        episodes_to_retry = [ep['episode_data'] for ep in failed_episodes]
        
        # Reuse existing parallel processing logic
        results = self.process_instructions_parallel(episodes_to_retry, encoder, max_workers)
        
        # Summarize results
        success_count = sum(1 for r in results if r.get('success', False))
        total_count = len(results)
        
        print(f"\n=== Reprocessing completed ===")
        print(f"Success: {success_count}/{total_count}")
        print(f"Failed: {total_count - success_count}/{total_count}")
        
        # Save successful graphs by scene
        if success_count > 0:
            print(f"\n--- Saving {len(self.graphs)} successful graphs by scene ---")
            self.save_graphs_by_scene()
            print("Reprocessed graphs have been merged into existing files!")
        
        # Save still failed episodes to new file
        if self.failed_episodes:
            self._save_failed_episodes(self.failed_episodes)
            print(f"\nSaved {len(self.failed_episodes)} episodes that still failed in reprocessing.")
        
        return {
            'success': True,
            'total': total_count,
            'success_count': success_count,
            'failed_count': total_count - success_count,
            'results': results
        }


# Usage examples and test functions
def test_prompt_based_encoder(use_openrouter: bool = False, model: str = None):
    """Test PromptBasedGraphEncoder"""
    print("=== Testing Prompt-based Graph Encoder ===")
    
    # 1. Create Encoder (read API key from environment variables)
    import os
    
    if use_openrouter:
        # Use OpenRouter
        api_key = os.getenv('OPENROUTER_API_KEY')
        model = model or "meta-llama/llama-3.1-8b-instruct"
        print(f"Using OpenRouter with model: {model}")
        encoder = PromptBasedGraphEncoder(api_key=api_key, model=model, use_openrouter=True)
    else:
        # Use OpenAI
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables")
            print("Please set OPENAI_API_KEY environment variable or pass api_key parameter")
            return
        model = model or "gpt-4o"
        print(f"Using OpenAI with model: {model}")
        encoder = PromptBasedGraphEncoder(api_key=api_key, model=model, use_openrouter=False)
    
    # Pass model information to manager
    manager = PromptBasedGraphManager(model_name=model, use_openrouter=use_openrouter)
    
    # 2. Load test episodes
    with open("/HabitatLLM/data/datasets/memory_acquisition_stage.json", "r") as f:
        episodes = json.load(f)['episodes']
    
    # 3. Process in batches of 201 in parallel
    batch_size = 201
    total_episodes = len(episodes)
    
    for batch_start in range(0, total_episodes, batch_size):
        batch_end = min(batch_start + batch_size, total_episodes)
        batch_episodes = episodes[batch_start:batch_end]
        
        print(f"\n=== Processing batch {batch_start//batch_size + 1}: episodes {batch_start+1}-{batch_end} (parallel processing) ===")
        
        # Execute parallel processing
        results = manager.process_instructions_parallel(batch_episodes, encoder, max_workers=201)
        
        # Summarize results
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful
        print(f"\nBatch {batch_start//batch_size + 1} completed: {successful} success, {failed} failed")
        print(f"Total graphs accumulated so far: {len(manager.graphs)}")
    
    print("\n=== All batches completed ===")
    
    # Save by scene at the end after all processing is complete
    print(f"\n--- Final save: grouping {len(manager.graphs)} graphs by scene ---")
    manager.save_graphs_by_scene()
    print("All graphs saved by scene!")
    
    # Save accumulated failed episodes only once at the end
    if manager.failed_episodes:
        manager._save_failed_episodes(manager.failed_episodes)
        print(f"\nSaved a total of {len(manager.failed_episodes)} failed episodes.")


def example_usage(use_openrouter: bool = False, model: str = None):
    """Usage example"""
    import os
    
    if use_openrouter:
        # Use OpenRouter
        api_key = os.getenv('OPENROUTER_API_KEY')
        model = model or "meta-llama/llama-3.1-8b-instruct"
        print(f"Using OpenRouter with model: {model}")
        encoder = PromptBasedGraphEncoder(api_key=api_key, model=model, use_openrouter=True)
    else:
        # Use OpenAI
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("Please set OPENAI_API_KEY environment variable")
            return
        model = model or "gpt-4o"
        print(f"Using OpenAI with model: {model}")
        encoder = PromptBasedGraphEncoder(api_key=api_key, model=model, use_openrouter=False)
    
    # 2. Process single instruction
    instruction = "Put my favorite coffee mug on the kitchen counter. It's a blue ceramic mug with a small chip on the handle."
    
    try:
        graph = encoder.encode_instruction(instruction)
        print(f"Generated graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
        # 3. Save graph (in model-specific folder)
        manager = PromptBasedGraphManager(model_name=model, use_openrouter=use_openrouter)
        result = manager.process_instruction(instruction, encoder, "example")
        
        if result["success"]:
            print(f"Success!")
            manager.save_graphs_by_scene()
        else:
            print(f"Failed: {result['error']}")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Prompt-based Graph Encoder')
    parser.add_argument('--mode', choices=['test', 'example', 'retry'], default='test', 
                       help='Run mode: test, example, or retry')
    parser.add_argument('--use_openrouter', action='store_true', default=False,
                       help='Use OpenRouter API instead of OpenAI')
    parser.add_argument('--model', type=str, 
                       help='Model name (default: gpt-4o for OpenAI, meta-llama/llama-3.1-8b-instruct for OpenRouter)')
    parser.add_argument('--failed_file', type=str, 
                       help='Path to failed episode file for reprocessing (for retry mode)')
    
    args = parser.parse_args()
    
    if args.mode == "example":
        example_usage(use_openrouter=args.use_openrouter, model=args.model)
    elif args.mode == "retry":
        if not args.failed_file:
            print("--failed_file option is required in retry mode.")
            sys.exit(1)
        
        # Create Encoder
        import os
        api_key = os.getenv("OPENROUTER_API_KEY") if args.use_openrouter else os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("API key not found. Please check environment variables.")
            sys.exit(1)
        
        model = args.model or ("meta-llama/llama-3.1-8b-instruct" if args.use_openrouter else "gpt-4o")
        encoder = PromptBasedGraphEncoder(api_key=api_key, model=model, use_openrouter=args.use_openrouter)
        
        # Create Manager and reprocess (pass model information)
        manager = PromptBasedGraphManager(model_name=model, use_openrouter=args.use_openrouter)
        result = manager.retry_failed_episodes(args.failed_file, encoder)
        
        if result['success']:
            print(f"\nReprocessing result: {result['success_count']}/{result['total']} success")
        else:
            print(f"Reprocessing failed: {result['error']}")
    else:
        test_prompt_based_encoder(use_openrouter=args.use_openrouter, model=args.model)
