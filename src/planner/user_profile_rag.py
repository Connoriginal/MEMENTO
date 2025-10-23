#!/usr/bin/env python3

import json
import os
import re
from typing import List, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util


class UserProfileRAG:
    """
    Semantic RAG class that loads from user_profile_memory.json
    """
    def __init__(self, user_profile_memory_path=None, scene_id=None, ensure_same_scene=True, llm=None, openai_api_key=None):
        self._device = "cuda"
        # Configurable path from config
        if user_profile_memory_path is None:
            self.user_profile_memory_path = "/HabitatLLM/rebuttal/structurize/test_output/scenes"  # default value
            #TODO: need to add semantic memory path
        else:
            self.user_profile_memory_path = user_profile_memory_path
        self.scene_id = scene_id
        self.ensure_same_scene = ensure_same_scene
        
        # LLM setup (use LLM passed from llm_planner)
        self.llm = llm
        self.openai_api_key = openai_api_key
        
        # Same structure as existing RAG
        self.data_dict = {}
        self.index = 0
        self.episode_retrieval_info = {}
        self.episode_rerank_info = {}
        
        # Load data and generate embeddings
        self.load_user_profile_data()
        self.build_data_embedding()
    
    def _llm_generate(self, prompt: str) -> str:
        """Wrapper method for LLM calls (rag.py style)"""
        if self.llm is None:
            print("Warning: LLM is not available. Returning empty string.")
            return ""
            
        try:
            # Call in rag.py generate method style
            response = self.llm.generate(prompt)
            return response.strip()
        
        except Exception as e:
            print(f"Error occurred during LLM call: {e}")
            return ""

    def load_user_profile_data(self):
        """Load data from graph-structured JSON"""
        self.user_profile_memory_path = os.path.join(self.user_profile_memory_path, f"scene_{self.scene_id}_graphs.json")
        print(f"Loading user profile memory from {self.user_profile_memory_path}")
        
        with open(self.user_profile_memory_path, 'r') as f:
            scene_data = json.load(f)
        
        # Process each graph from "graphs" array
        for graph in scene_data.get("graphs", []):
            # Scene filtering (same logic as existing RAG)
            if self.ensure_same_scene and self.scene_id:
                if graph.get("scene_id") != self.scene_id:
                    continue
            
            # Extract information from graph
            knowledge_info = self._extract_knowledge_from_graph(graph)
            
            info = {
                "instruction": graph.get("instruction", ""),
                "episode_id": int(graph.get("episode_id", 0)),
                "scene_id": graph.get("scene_id", ""),
                "knowledge": knowledge_info["knowledge"],  # key (embedding target)
                "descriptions": knowledge_info["descriptions"],  # value (return in natural language)
                "knowledge_type": knowledge_info["knowledge_type"],  # object_semantics or user_pattern
                "graph": graph,  # preserve original graph structure
                "file": self.user_profile_memory_path
            }
            
            self.data_dict[self.index] = info
            self.index += 1

        print(f"Loaded {len(self.data_dict)} user profile memories from {scene_data.get('total_graphs', 0)} graphs")
        
        # Check knowledge type distribution
        type_counts = {"object_semantics": 0, "user_pattern": 0}
        for idx in self.data_dict:
            ktype = self.data_dict[idx].get("knowledge_type", "unknown")
            if ktype in type_counts:
                type_counts[ktype] += 1
        
        print(f"Knowledge type distribution: {type_counts}")
        if type_counts["user_pattern"] <= 5:
            print(f"[WARNING] Very few user_pattern memories: {type_counts['user_pattern']}")
    
    def _extract_knowledge_from_graph(self, graph):
        """Extract knowledge information from graph structure"""
        knowledge_nodes = []
        object_nodes = []
        location_nodes = []
        descriptions = []
        knowledge_type = "object_semantics"  # default value
        
        # Analyze nodes
        for node in graph.get("nodes", []):
            if node.get("type") == "Knowledge":
                knowledge_nodes.append(node)
                # determine knowledge type
                if node.get("subtype") == "user_pattern":
                    knowledge_type = "user_pattern"
                # add alias information
                if node.get("alias"):
                    descriptions.append(f"Knowledge: {node['alias']}")
            elif node.get("type") == "Object":
                object_nodes.append(node)
                # add object information to descriptions
                obj_desc = f"Object: {node.get('name', '')}"
                if node.get("attributes"):
                    attributes = ", ".join(node["attributes"])
                    obj_desc += f" ({attributes})"
                descriptions.append(obj_desc)
            elif node.get("type") == "Location":
                location_nodes.append(node)
                # add location information to descriptions
                loc_desc = f"Location: {node.get('name', '')}"
                if node.get("expression"):
                    loc_desc += f" - {node['expression']}"
                descriptions.append(loc_desc)
        
        # analyze edge relationships to generate additional descriptions
        edge_descriptions = self._extract_relationships_from_edges(graph.get("edges", []), graph.get("nodes", []))
        descriptions.extend(edge_descriptions)
        
        # generate knowledge text (for embedding)
        knowledge_texts = []
        for k_node in knowledge_nodes:
            if k_node.get("alias"):
                knowledge_texts.append(k_node["alias"])
        
        # use instruction if no knowledge available
        if not knowledge_texts:
            knowledge_texts.append(graph.get("instruction", ""))
        
        return {
            "knowledge": knowledge_texts,
            "descriptions": descriptions,
            "knowledge_type": knowledge_type
        }
    
    def _extract_relationships_from_edges(self, edges, nodes):
        """Extract relationship information from edges"""
        descriptions = []
        
        # create dictionary mapping node IDs to node information
        node_map = {node["id"]: node for node in nodes}
        
        for edge in edges:
            source_id = edge.get("source_id")
            target_id = edge.get("target_id")
            relation = edge.get("relation")
            
            source_node = node_map.get(source_id)
            target_node = node_map.get(target_id)
            
            if source_node and target_node and relation:
                source_name = self._get_node_name(source_node)
                target_name = self._get_node_name(target_node)
                
                # describe relationships in natural language
                if relation == "alias_of":
                    descriptions.append(f"{source_name} is an alias of {target_name}")
                elif relation == "next_to":
                    descriptions.append(f"{source_name} is next to {target_name}")
                elif relation == "target_object":
                    descriptions.append(f"{source_name} targets {target_name}")
                elif relation == "target_location":
                    descriptions.append(f"{source_name} targets location {target_name}")
                elif relation == "owns":
                    descriptions.append(f"{source_name} owns {target_name}")
                elif relation == "entails":
                    descriptions.append(f"{source_name} entails {target_name}")
                elif relation == "before":
                    descriptions.append(f"{source_name} happens before {target_name}")
                else:
                    descriptions.append(f"{source_name} {relation} {target_name}")
        
        return descriptions
    
    def _get_node_name(self, node):
        """Extract display name from node"""
        if node.get("type") == "User":
            return node.get("name", "User")
        elif node.get("type") == "Knowledge":
            return node.get("alias", "Knowledge")
        elif node.get("type") == "Object":
            return node.get("name", "Object")
        elif node.get("type") == "Location":
            return node.get("name", "Location")
        else:
            return node.get("id", "Unknown")
    
    def build_data_embedding(self):
        """Use only personalized_knowledge (alias) as embedding target"""
        self.embedding_model = SentenceTransformer(
            model_name_or_path="all-mpnet-base-v2", device=self._device
        )

        # use only personalized_knowledge (alias) as embedding target
        knowledge_list = []
        self.knowledge_to_index = {}  # alias -> data_dict index mapping
        
        for index in self.data_dict:
            memory = self.data_dict[index]
            # knowledge is alias list, so use first alias
            if memory["knowledge"] and len(memory["knowledge"]) > 0:
                alias = memory["knowledge"][0]  # use first alias
            else:
                alias = memory["instruction"]  # fallback
            
            knowledge_list.append(alias)
            self.knowledge_to_index[len(knowledge_list) - 1] = index

        print(f"Creating embeddings for {len(knowledge_list)} personalized knowledge entries")
        for i, knowledge in enumerate(knowledge_list):
            print(f"  {i}: {knowledge}")

        # generate knowledge embeddings
        self.knowledge_embeddings = self.embedding_model.encode(
            knowledge_list, batch_size=32, convert_to_tensor=True,
        )
        
        print(f"Generated embeddings shape: {self.knowledge_embeddings.shape}")
        
    
    def extract_personalized_info(self, query: str) -> dict:
        """
        Extract personalized information through LLM query analysis - Option 1 structure
        
        Returns:
            {
                "object_semantics": ["grandmother's vase", "favorite coffee mug"],
                "user_pattern": ["dinner party setup routine"]
            }
        """
        
        prompt = f"""Analyze the following instruction and extract personalized information by type:
        
        Extract personalized items and group them by knowledge type:
        - object_semantics: specific objects, items, places with personal meaning/attributes
        - user_pattern: behavioral patterns, routines, setups, arrangements, preferences
        
        Output format (use empty arrays if no items for that type):
        {{
            "object_semantics": [list of personalized objects/places],
            "user_pattern": [list of behavioral patterns/routines]
        }}
        
        [Example 1]
        Instruction: Please put the ceramic bowl, the wooden cutting board, and my mug from my grandmother back on the kitchen counter.
        Output: 
        {{
            "object_semantics": ["mug from my grandmother"],
            "user_pattern": []
        }}
        
        [Example 2]
        Instruction: Put my decorative collection together in another shelf in the living room and place them next to each other.
        Output: 
        {{
            "object_semantics": ["my decorative collection"],
            "user_pattern": []
        }}
        
        [Example 3]
        Instruction: Can you place my book for bedtime reading?
        Output: 
        {{
            "object_semantics": [],
            "user_pattern": ["bedtime reading setup"]
        }}
        
        [Example 4]
        Instruction: Can you set up the calming atmosphere in the bedroom?
        Output: 
        {{
            "object_semantics": [],
            "user_pattern": ["calming atmosphere in the bedroom setup"]
        }}
        
        [Example 5]
        Instruction: Can you organize the kitchenware for easy table setting?
        Output: 
        {{
            "object_semantics": [],
            "user_pattern": ["kitchenware organization for easy table setting"]
        }}
        
        [Example 6 - Joint Task]
        Instruction: Could you move the book gifted by my dear friend and the Olive Kids Paisley Pencil Case from the bedroom to the living room and place them on the table? Then, could you move my personal work device and the black mouse pad with the logo from the office table to the bedroom chest of drawers? Place them next to each other on the chest of drawers.
        Output: 
        {{
            "object_semantics": ["gifted book by my dear friend", "my personal work device"],
            "user_pattern": []
        }}
        
        [Example 7 - Joint Task]
        Instruction: Place the black and orange sports shoe and the pencil case from my childhood friend on the bedroom chest of drawers. Then, could you move my decorative set and the Cole Hardware Orchid Pot 85 from the dining table to the living room couch and place them next to each other?
        Output: 
        {{
            "object_semantics": ["the pencil case from my childhood friend", "my decorative set"],
            "user_pattern": []
        }}
        
        [Example 8 - Joint Task]
        Instruction: Move my work setup to the office table and place them next to each other. After that, can you set up my late-night work session area?
        Output: 
        {{
            "object_semantics": ["my work setup"],
            "user_pattern": ["late-night work session area setup"]
        }}
        
        [Example 9 - Joint Task]
        Instruction: Could you set up the shelves by moving the picture frame with my family reunion photo and the lamp with a pineapple-shaped neon outline and a black base? After that, could you set the relaxing ambiance in the bedroom for me?
        Output: 
        {{
            "object_semantics": ["the picture frame with my family reunion photo"],
            "user_pattern": ["relaxing ambiance in the bedroom setup"]
        }}
        
        [Example 10 - Joint Task]
        Instruction: Can you set up my study materials for the afternoon session? Additionally, can you set up my evening work spot?
        Output: 
        {{
            "object_semantics": [],
            "user_pattern": ["afternoon session study materials setup", "evening work spot setup"]
        }}
        
        [Example 11 - Joint Task]
        Instruction: Could you set up the cozy evening atmosphere for me? Additionally, can you store the jug and cup where I usually keep them for breakfast?
        Output: 
        {{
            "object_semantics": [],
            "user_pattern": ["cozy evening atmosphere setup", "jug and cup storage for breakfast setup"]
        }}
        
        Instruction: {query}
        Output:
        
        """
        
        # LLM call and parsing
        response = self._llm_generate(prompt)
        # JSON parsing
        def extract_json_from_response(response_text):
            """Extract JSON from LLM response"""
            # 1. remove ```json code blocks
            if "```json" in response_text:
                pattern = r'```json\s*(.*?)\s*```'
                match = re.search(pattern, response_text, re.DOTALL)
                if match:
                    return match.group(1).strip()
                else:
                    # extract only content after ```json
                    start_idx = response_text.find("```json") + 7
                    return response_text[start_idx:].strip()
            
            # 2. remove ``` general code blocks
            elif response_text.startswith("```") and response_text.endswith("```"):
                return response_text[3:-3].strip()
            
            # 3. extract JSON between { }
            elif "{" in response_text and "}" in response_text:
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}") + 1
                return response_text[start_idx:end_idx].strip()
            
            # 4. return as is
            return response_text.strip()
        
        try:
            cleaned_response = extract_json_from_response(response)
            parsed_info = json.loads(cleaned_response)
            return parsed_info
        except json.JSONDecodeError:
            return {
                "object_semantics": ["my coffee mug"],
                "user_pattern": []
            }
    
    
    def filter_graph_with_personalized_info(self, personalized_info: dict) -> dict:
        """
        Filter graphs by type according to personalized information
        
        Returns:
            {
                "object_semantics": [list of filtered indices],
                "user_pattern": [list of filtered indices]
            }
        """
        filtered_results = {
            "object_semantics": [],
            "user_pattern": []
        }
        
        # extract knowledge by type from personalized_info
        object_items = personalized_info.get("object_semantics", [])
        pattern_items = personalized_info.get("user_pattern", [])
        
        # filter by each type
        for knowledge_type in ["object_semantics", "user_pattern"]:
            for index in self.data_dict:
                memory = self.data_dict[index]
                
                # check if memory's knowledge_type matches current type
                if memory.get("knowledge_type") == knowledge_type:
                    filtered_results[knowledge_type].append(index)
        
        # log output
        total_object = len(filtered_results["object_semantics"])
        total_pattern = len(filtered_results["user_pattern"])
        print(f"Filtered memories: object_semantics={total_object}, user_pattern={total_pattern}")
        print(f"Object items to search: {object_items}")
        print(f"Pattern items to search: {pattern_items}")
        
        return filtered_results
    
    def _retrieve_for_knowledge_type(self, knowledge_text: str, filtered_indices: list, knowledge_type: str) -> list:
        """
        Perform retrieval for specific knowledge type
        
        Returns:
            list of (score, actual_idx, knowledge_type) tuples
        """
        if not filtered_indices:
            return []
        
        # knowledge_text embedding
        query_embedding = self.embedding_model.encode(knowledge_text, convert_to_tensor=True)
        
        # collect embeddings corresponding to filtered indices
        filtered_embeddings = []
        filtered_to_actual = {}  # filtered result index -> actual data_dict index
        
        for i, actual_idx in enumerate(filtered_indices):
            # find embedding index corresponding to actual_idx
            for emb_idx, data_idx in self.knowledge_to_index.items():
                if data_idx == actual_idx:
                    filtered_embeddings.append(self.knowledge_embeddings[emb_idx])
                    filtered_to_actual[len(filtered_embeddings) - 1] = actual_idx
                    break
        
        if not filtered_embeddings:
            print(f"   [Warning]: No embeddings found for {knowledge_type}")
            return []
        
        filtered_embeddings = torch.stack(filtered_embeddings)
        
        # calculate similarity
        dot_scores = util.dot_score(query_embedding, filtered_embeddings)[0]
        scores = dot_scores.cpu().numpy()
        
        # format results
        results = []
        for i, score in enumerate(scores):
            actual_idx = filtered_to_actual[i]
            results.append((float(score), actual_idx, knowledge_type))
        
        # sort by score
        results.sort(key=lambda x: x[0], reverse=True)
        
        print(f"   Found {len(results)} candidates for '{knowledge_text}' in {knowledge_type}")
        return results
    
    def graph_to_natural_language(self, graph: dict) -> str:
        """
        Summarize graph in natural language form
        """
        
        prompt = f"""Analyze the following graph and interpret as natural language.
        FOCUS on the user's personalized knowledge and what it is and how we can use it to understand the instruction.
        Provide only a simple text description without any formatting or JSON.
        Also provide only the output with any additional explanation.
        
        For example:

        [Example 1]
        
        Graph:
        {{
            "knowledge_type": "object_semantics",
            "nodes": [
            {{ "id": "u1", "type": "User", "name": "user_A" }},
            {{ "id": "k1", "type": "Knowledge", "subtype": "object_semantics", "alias": "gift book from friend" }},
            {{ "id": "o1", "type": "Object", "name": "book", "granularity": "instance", "attributes": ["white", "subtle yellow accents", "bookmark"], "provenance": "gift from friend" }}
            ],
            "edges": [
            {{ "source": "u1", "target": "k1", "type": "Hierarchical", "relation": "owns" }},
            {{ "source": "k1", "target": "o1", "type": "Hierarchical", "relation": "alias_of" }}
            ]
        }}
        Output: The user's gift book from friend is a white book with subtle yellow accents and a bookmark.


        [Example 2]
        
        Graph:
        {{
            "knowledge_type": "user_pattern",
            "nodes": [
            {{ "id": "u1", "type": "User", "name": "user_A" }},
            {{ "id": "k1", "type": "Knowledge", "subtype": "user_pattern", "alias": "bedtime reading routine", "description": "keep the book on the chest of drawers in the bedroom for bedtime reading" }},
            {{ "id": "o1", "type": "Object", "name": "book" }},
            {{ "id": "l1", "type": "Location", "name": "chest_of_drawers", "expression": "on the chest of drawers in the bedroom" }}
            ],
            "edges": [
            {{ "source": "u1", "target": "k1", "type": "Hierarchical", "relation": "owns" }},
            {{ "source": "k1", "target": "o1", "type": "Hierarchical", "relation": "target_object" }},
            {{ "source": "k1", "target": "l1", "type": "Hierarchical", "relation": "target_location", "object": "o1" }}
            ]
        }}
        Output: 
        {{ 
            "descriptions": "The user's bedtime reading routine is to keep the book on the chest of drawers in the bedroom for bedtime reading." 
        }}


        [Example 3]

        Graph:
        {{
            "knowledge_type": "object_semantics",
            "nodes": [
            {{ "id": "u1", "type": "User", "name": "user_A" }},
            {{ "id": "k1", "type": "Knowledge", "subtype": "object_semantics", "alias": "dinnerware set" }},
            {{ "id": "o1", "type": "Object", "name": "plate", "granularity": "instance", "attributes": ["square", "white", "black geometric pattern"] }},
            {{ "id": "o2", "type": "Object", "name": "bowl", "granularity": "instance", "attributes": ["white", "brown lid"] }}
            ],
            "edges": [
            {{ "source": "u1", "target": "k1", "type": "Hierarchical", "relation": "owns" }},
            {{ "source": "k1", "target": "o1", "type": "Hierarchical", "relation": "alias_of" }},
            {{ "source": "k1", "target": "o2", "type": "Hierarchical", "relation": "alias_of" }},
            {{ "source": "o1", "target": "o2", "type": "Hierarchical", "relation": "next_to" }}
            ]
        }}
        Output: The user's dinnerware set is a square white plate with a black geometric pattern and a white bowl with a brown lid.


        [Example 4]
        
        Graph:
        {{
            "knowledge_type": "user_pattern",
            "nodes": [
            {{ "id": "u1", "type": "User", "name": "user_A" }},

            {{
                "id": "k1",
                "type": "Knowledge",
                "subtype": "user_pattern",
                "alias": "living room cleaning setup",
                "description": "keep vase and stuffed toy together on the couch during cleaning for easy access later"
            }},

            {{ "id": "o1", "type": "Object", "name": "vase" }},
            {{ "id": "o2", "type": "Object", "name": "stuffed toy" }},

            {{ "id": "l1", "type": "Location", "name": "couch", "expression": "on the couch in the living room" }}
            ],
            "edges": [
            {{ "source": "u1", "target": "k1", "type": "Hierarchical", "relation": "owns" }},

            {{ "source": "k1", "target": "o1", "type": "Hierarchical", "relation": "target_object" }},
            {{ "source": "k1", "target": "l1", "type": "Hierarchical", "relation": "target_location", "object": "o1" }},

            {{ "source": "k1", "target": "o2", "type": "Hierarchical", "relation": "target_object" }},
            {{ "source": "k1", "target": "l1", "type": "Hierarchical", "relation": "target_location", "object": "o2" }},

            {{ "source": "o1", "target": "o2", "type": "Hierarchical", "relation": "next_to" }}
            ]
        }}
        Output: The user's living room cleaning setup is to keep vase and stuffed toy together on the couch during cleaning for easy access later.


        [Example 5]
        
        Graph:
        {{
            "knowledge_type": "user_pattern",
            "nodes": [
            {{ "id": "u1", "type": "User", "name": "user_A" }},

            {{
                "id": "k1",
                "type": "Knowledge",
                "subtype": "user_pattern",
                "alias": "living room refresh setup",
                "description": "rearrange vase, candle, and bowl to keep the living room setting fresh and inviting"
            }},

            {{ "id": "o1", "type": "Object", "name": "vase" }},
            {{ "id": "o2", "type": "Object", "name": "candle" }},
            {{ "id": "o3", "type": "Object", "name": "bowl" }},

            {{ "id": "l1", "type": "Location", "name": "livingroom_table", "expression": "different table in the living room" }}
            ],
            "edges": [
            {{ "source": "u1", "target": "k1", "type": "Hierarchical", "relation": "owns" }},

            {{ "source": "k1", "target": "o1", "type": "Hierarchical", "relation": "target_object" }},
            {{ "source": "k1", "target": "l1", "type": "Hierarchical", "relation": "target_location", "object": "o1" }},

            {{ "source": "k1", "target": "o2", "type": "Hierarchical", "relation": "target_object" }},
            {{ "source": "k1", "target": "l1", "type": "Hierarchical", "relation": "target_location", "object": "o2" }},

            {{ "source": "k1", "target": "o3", "type": "Hierarchical", "relation": "target_object" }},
            {{ "source": "k1", "target": "l1", "type": "Hierarchical", "relation": "target_location", "object": "o3" }}
            ]
        }}
        Output: The user's living room refresh setup is to rearrange vase, candle, and bowl to keep the living room setting fresh and inviting.


        [Example 6]
        
        Graph:
        {{
            "knowledge_type": "user_pattern",
            "nodes": [
            {{ "id": "u1", "type": "User", "name": "user_A" }},

            {{
                "id": "k1",
                "type": "Knowledge",
                "subtype": "user_pattern",
                "alias": "living room play setup",
                "description": "keep toy vehicle on the living room couch so it's easy to find when playing"
            }},

            {{ "id": "a1", "type": "Pattern", "name": "move", "args": ["toy vehicle", "from bedroom", "to living room"] }},
            {{ "id": "a2", "type": "Pattern", "name": "place", "args": ["toy vehicle", "on couch", "in living room"] }},

            {{ "id": "o1", "type": "Object", "name": "toy vehicle" }},
            {{ "id": "l1", "type": "Location", "name": "bedroom", "expression": "in the bedroom" }},
            {{ "id": "l2", "type": "Location", "name": "livingroom", "expression": "in the living room" }},
            {{ "id": "l3", "type": "Location", "name": "couch", "expression": "on the couch in the living room" }}
            ],
            "edges": [
            {{ "source": "u1", "target": "k1", "type": "Hierarchical", "relation": "owns" }},

            {{ "source": "k1", "target": "a1", "type": "Hierarchical", "relation": "entails" }},
            {{ "source": "k1", "target": "a2", "type": "Hierarchical", "relation": "entails" }},

            {{ "source": "a1", "target": "o1", "type": "Hierarchical", "relation": "target" }},
            {{ "source": "a1", "target": "l1", "type": "Hierarchical", "relation": "source_location" }},
            {{ "source": "a1", "target": "l2", "type": "Hierarchical", "relation": "target_location" }},

            {{ "source": "a2", "target": "o1", "type": "Hierarchical", "relation": "target" }},
            {{ "source": "a2", "target": "l3", "type": "Hierarchical", "relation": "target_location" }},

            {{ "source": "a1", "target": "a2", "type": "Temporal", "relation": "before" }}
            ]
        }}
        Output: The user's living room play setup is to keep toy vehicle on the living room couch so it's easy to find when playing.


        Graph: {graph}
        Output: 
        

        
        """
        
        # LLM call - simple text response
        response = self._llm_generate(prompt)
        print(f"LLM Response: {response[:100]}...")  # log for response verification
        
        # simply return text response as is
        return response.strip()
    
    
    def retrieve_top_k_given_query(self, query: str, top_k: int = 5, agent_id: int = 0, related_episode_id: Union[List[int], int] = []):
        assert query != "", "query text is an empty string"
        
        available_memories = len(self.data_dict)
        if top_k > available_memories:
            print(f"   [Warning]: top_k ({top_k}) > available memories ({available_memories}), adjusting to {available_memories}")
            top_k = available_memories
        
        if available_memories == 0:
            print(f"   [Warning]: No user profile memories available for this scene")
            return np.array([]), np.array([])
        
        # 1. extract personalized info using LLM (Option 1 structure)
        personalized_info = self.extract_personalized_info(query)
        object_items = personalized_info.get("object_semantics", [])
        pattern_items = personalized_info.get("user_pattern", [])
        
        # 2. filter graphs by type
        filtered_results = self.filter_graph_with_personalized_info(personalized_info)
        
        # 3. unified processing with (knowledge, type) pairs
        knowledge_type_pairs = []
        
        # add object_semantics items
        for item in object_items:
            knowledge_type_pairs.append((item, "object_semantics"))
        
        # add user_pattern items  
        for item in pattern_items:
            knowledge_type_pairs.append((item, "user_pattern"))
        
        print(f"\nProcessing {len(knowledge_type_pairs)} knowledge-type pairs:")
        for knowledge, ktype in knowledge_type_pairs:
            print(f"  - '{knowledge}' ({ktype})")
        
        # process each (knowledge, type) pair to collect top-5 each
        all_results = []  # [(score, actual_idx, knowledge_type), ...]
        
        for knowledge_text, knowledge_type in knowledge_type_pairs:
            print(f"\n--- Retrieving for '{knowledge_text}' ({knowledge_type}) ---")
            
            # get filtered indices for the corresponding type
            filtered_indices = filtered_results.get(knowledge_type, [])
            
            if not filtered_indices:
                print(f"   No filtered memories for {knowledge_type}")
                continue
            
            # perform retrieval for the corresponding knowledge-type pair
            pair_results = self._retrieve_for_knowledge_type(
                knowledge_text=knowledge_text,
                filtered_indices=filtered_indices,
                knowledge_type=knowledge_type
            )
            
            # select top-5 from each pair
            top5_results = pair_results[:5]  # already sorted
            all_results.extend(top5_results)
            
            print(f"   Selected top-{len(top5_results)} for '{knowledge_text}'")
        
        # 3.5. remove duplicates (keep highest score based on index)
        unique_results = {}  # actual_idx -> (best_score, knowledge_type, source_knowledge)
        for score, actual_idx, knowledge_type in all_results:
            if actual_idx not in unique_results or score > unique_results[actual_idx][0]:
                # track which knowledge the result came from
                source_knowledge = None
                for knowledge_text, ktype in knowledge_type_pairs:
                    if ktype == knowledge_type:
                        source_knowledge = knowledge_text
                        break
                unique_results[actual_idx] = (score, knowledge_type, source_knowledge)
        
        print(f"\nAfter deduplication: {len(unique_results)} unique memories (from {len(all_results)} total)")
        
        # sort final results (by score)
        final_results = sorted(unique_results.items(), key=lambda x: x[1][0], reverse=True)
        
        # 4. if no results, search from entire dataset (fallback)
        if not final_results:
            print("No type-specific results found, falling back to full search")
            all_indices = list(self.data_dict.keys())
            fallback_query = query  # use entire query
            fallback_results = self._retrieve_for_knowledge_type(
                knowledge_text=fallback_query,
                filtered_indices=all_indices,
                knowledge_type="fallback"
            )
            # select only top-5 from fallback as well
            fallback_sorted = sorted(fallback_results, key=lambda x: x[0], reverse=True)
            fallback_top5 = fallback_sorted[:5]
            
            # convert fallback results to final_results format
            final_results = []
            for score, actual_idx, knowledge_type in fallback_top5:
                final_results.append((actual_idx, (score, knowledge_type, "fallback")))
            
            print(f"Using {len(final_results)} fallback results")
        
        # 5. format final results (after deduplication)
        final_scores = [score for _, (score, _, _) in final_results]
        actual_indices = [actual_idx for actual_idx, _ in final_results]
        
        print(f"\nFinal deduplicated results ({len(final_results)} total):")
        for i, (actual_idx, (score, knowledge_type, source_knowledge)) in enumerate(final_results):
            episode_id = self.data_dict[actual_idx].get('episode_id', 'unknown')
            print(f"  [{i+1}] Episode {episode_id}, Score: {score:.4f}, Type: {knowledge_type}, Source: '{source_knowledge}'")
        
        # 8. natural language conversion and existing logic
        used_episodes = [self.data_dict[actual_idx]["episode_id"] for actual_idx in actual_indices]
        accuracy = self.calculate_accuracy(used_episodes, related_episode_id)
        
        # summarize retrieved graphs to natural language form using LLM
        natural_language_results = []
        
        for i, actual_idx in enumerate(actual_indices):
            memory = self.data_dict[actual_idx]
            try:
                # convert each graph to natural language (text response)
                nl_result = self.graph_to_natural_language(memory)
                natural_language_results.append({
                    'index': i,
                    'actual_idx': actual_idx,
                    'episode_id': memory.get('episode_id', 'unknown'),
                    'natural_language': nl_result,
                    'original_memory': memory
                })
                print(f"  [{i+1}] Episode {memory.get('episode_id', 'unknown')}: {nl_result[:100]}...")
            except Exception as e:
                print(f"  [{i+1}] Failed to convert episode {memory.get('episode_id', 'unknown')}: {str(e)}")
                natural_language_results.append({
                    'index': i,
                    'actual_idx': actual_idx,
                    'episode_id': memory.get('episode_id', 'unknown'),
                    'natural_language': f"Error: {str(e)}",
                    'original_memory': memory
                })

        # update data_dict descriptions with natural language results (for llm_planner to use)
        for result in natural_language_results:
            actual_idx = result['actual_idx']
            nl_text = result['natural_language']
            
            # backup original descriptions (only once)
            if 'original_descriptions' not in self.data_dict[actual_idx]:
                self.data_dict[actual_idx]['original_descriptions'] = self.data_dict[actual_idx]['descriptions']
            
            # update descriptions to natural language
            self.data_dict[actual_idx]['descriptions'] = nl_text
        
        # logging
        current_episode_id = getattr(self, 'current_episode_id', None)
        if current_episode_id is not None:
            if not hasattr(self, 'episode_retrieval_info'):
                self.episode_retrieval_info = {}
            self.episode_retrieval_info[current_episode_id] = {
                'used_episodes': str(used_episodes),
                'related_episode_id': str(related_episode_id) if related_episode_id else "[]",
                'accuracy': accuracy
            }
            print(f"Stored user profile retrieval info for episode {current_episode_id}: {self.episode_retrieval_info[current_episode_id]}")
        else:
            print(f"Warning: current_episode_id is None, cannot store user profile retrieval info")

        return np.array(final_scores), actual_indices
    
    
    def calculate_accuracy(self, used_episodes, related_episode_ids):
        """Calculate accuracy using the same logic as the original RAG."""
        accuracy = 0
        
        if isinstance(related_episode_ids, int) or isinstance(related_episode_ids, str):
            if int(related_episode_ids) in used_episodes:
                accuracy = 1
        elif isinstance(related_episode_ids, list):
            for ep_id in related_episode_ids:
                if int(ep_id) in used_episodes:
                    accuracy += 1
            accuracy /= len(related_episode_ids)
        
        return accuracy

    def save_retrieval_logs(self, output_dir, include_rerank=True):
        """Save user profile RAG retrieval and rerank logs as CSV files"""
        import pandas as pd
        import time
        
        episode_retrieval_info = getattr(self, 'episode_retrieval_info', {})
        episode_rerank_info = getattr(self, 'episode_rerank_info', {})
        
        # Save user profile retrieval logs
        if episode_retrieval_info:
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, "user_profile_retrieval_logs.csv")
            lock_path = os.path.join(output_dir, "user_profile_retrieval_logs.lock")
            
            # Use file lock to prevent concurrent access
            max_retries = 10
            retry_delay = 0.5
            
            for attempt in range(max_retries):
                try:
                    # Try to acquire lock
                    if os.path.exists(lock_path):
                        # Check if lock is stale (older than 30 seconds)
                        lock_age = time.time() - os.path.getmtime(lock_path)
                        if lock_age > 30:
                            print(f"Removing stale user profile lock file (age: {lock_age:.1f}s)")
                            os.remove(lock_path)
                    
                    # Create lock file
                    with open(lock_path, 'w') as f:
                        f.write(str(os.getpid()))
                    
                    # Check if existing CSV file exists
                    existing_df = None
                    if os.path.exists(csv_path):
                        try:
                            existing_df = pd.read_csv(csv_path)
                            print(f"Found existing user profile CSV file with {len(existing_df)} entries")
                        except Exception as e:
                            print(f"Error reading existing user profile CSV: {e}")
                            existing_df = None
                    
                    # Convert dictionary to DataFrame format
                    log_data = []
                    for episode_id, info in episode_retrieval_info.items():
                        log_data.append({
                            'episode_id': episode_id,
                            'used_episodes': info['used_episodes'],
                            'related_episode_id': info['related_episode_id'],
                            'accuracy': info['accuracy']
                        })
                    
                    new_df = pd.DataFrame(log_data)
                    
                    # Combine with existing data if available
                    if existing_df is not None:
                        # Remove duplicates based on episode_id
                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                        combined_df = combined_df.drop_duplicates(subset=['episode_id'], keep='last')
                        print(f"Combined {len(existing_df)} existing + {len(new_df)} new user profile entries = {len(combined_df)} total")
                    else:
                        combined_df = new_df
                        print(f"Created new user profile CSV with {len(combined_df)} entries")
                    
                    combined_df.to_csv(csv_path, index=False)
                    print(f"User profile retrieval logs saved to: {csv_path}")
                    print(f"Total user profile retrieval entries: {len(combined_df)}")
                    
                    # Remove lock file
                    if os.path.exists(lock_path):
                        os.remove(lock_path)
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    print(f"User profile save attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 1.5  # Exponential backoff
                    else:
                        print(f"Failed to save user profile retrieval logs after {max_retries} attempts")
                        # Clean up lock file if it exists
                        if os.path.exists(lock_path):
                            try:
                                os.remove(lock_path)
                            except:
                                pass
        else:
            print("No user profile retrieval info to save")
        
        # Save user profile rerank logs if available
        if include_rerank and episode_rerank_info:
            rerank_csv_path = os.path.join(output_dir, "user_profile_rerank_logs.csv")
            rerank_lock_path = os.path.join(output_dir, "user_profile_rerank_logs.lock")
            
            # Similar logic for rerank logs...
            print(f"User profile rerank logs would be saved to: {rerank_csv_path}")
            print(f"Total user profile rerank entries: {len(episode_rerank_info)}")
        elif include_rerank:
            print("No user profile rerank info to save")