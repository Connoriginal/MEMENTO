"""
Semantic Memory Graph Schema Definition

그래프 구조의 노드와 엣지 스키마만 정의
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum


# === Node Types ===
class NodeType(Enum):
    USER = "User"
    KNOWLEDGE = "Knowledge"
    PATTERN = "Pattern"
    OBJECT = "Object"
    LOCATION = "Location"


# === Edge Types ===
class EdgeRelation(Enum):
    OWNS = "owns"  # User -> Knowledge
    ENTAILS = "entails"  # Knowledge -> Pattern
    TARGET = "target"  # Pattern -> Object/Location
    ALIAS_OF = "alias_of"  # Knowledge -> Object
    BEFORE = "before"  # Pattern -> Pattern (temporal)
    NEXT_TO = "next_to"  # Object -> Object (spatial relation)
    TARGET_OBJECT = "target_object"  # Knowledge -> Object
    TARGET_LOCATION = "target_location"  # Knowledge -> Location
    SOURCE_LOCATION = "source_location"  # Pattern -> Location


class EdgeType(Enum):
    HIERARCHICAL = "Hierarchical"
    TEMPORAL = "Temporal"


# === Knowledge Subtypes ===
class KnowledgeSubtype(Enum):
    OBJECT_SEMANTICS = "object_semantics"
    USER_PATTERN = "user_pattern"


# === Object Granularity ===
class ObjectGranularity(Enum):
    TYPE = "type"
    INSTANCE = "instance"


# === Base Node Class ===
@dataclass
class GraphNode:
    id: str
    type: NodeType
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value
        }


# === Node Implementations ===

@dataclass
class UserNode(GraphNode):
    name: str
    
    def __post_init__(self):
        self.type = NodeType.USER
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "name": self.name
        }


@dataclass
class KnowledgeNode(GraphNode):
    subtype: KnowledgeSubtype
    alias: str
    description: str
    
    def __post_init__(self):
        self.type = NodeType.KNOWLEDGE
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            **super().to_dict(),
            "subtype": self.subtype.value,
            "alias": self.alias
        }
        if self.description:
            result["description"] = self.description
        return result


@dataclass
class PatternNode(GraphNode):
    name: str
    args: List[str]
    
    def __post_init__(self):
        self.type = NodeType.PATTERN
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "name": self.name,
            "args": self.args
        }


@dataclass
class ObjectNode(GraphNode):
    name: str
    granularity: ObjectGranularity
    attributes: List[str] = field(default_factory=list)
    provenance: str = None
    
    def __post_init__(self):
        self.type = NodeType.OBJECT
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            **super().to_dict(),
            "name": self.name,
            "granularity": self.granularity.value
        }
        if self.attributes:
            result["attributes"] = self.attributes
        if self.provenance:
            result["provenance"] = self.provenance
        return result


@dataclass
class LocationNode(GraphNode):
    name: str
    expression: str
    
    def __post_init__(self):
        self.type = NodeType.LOCATION
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "name": self.name,
            "expression": self.expression
        }


# === Edge Class ===
@dataclass
class GraphEdge:
    source: str
    target: str
    type: EdgeType
    relation: EdgeRelation
    object: str = None  # Optional object reference for target_location edges
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "source": self.source,
            "target": self.target,
            "type": self.type.value,
            "relation": self.relation.value
        }
        if self.object:
            result["object"] = self.object
        return result


# === Graph Container ===
@dataclass
class SemanticGraph:
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: List[GraphEdge] = field(default_factory=list)
    episode_id: str = ""
    scene_id: str = ""
    answer_knowledge_type: str = ""
    instruction: str = ""
    
    def add_node(self, node: GraphNode):
        self.nodes[node.id] = node
    
    def add_edge(self, edge: GraphEdge):
        self.edges.append(edge)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges]
        }
        if self.episode_id:
            result["episode_id"] = self.episode_id
        if self.scene_id:
            result["scene_id"] = self.scene_id
        if self.instruction:
            result["instruction"] = self.instruction
        if self.answer_knowledge_type:
            result["answer_knowledge_type"] = self.answer_knowledge_type
        return result


# === Example Schema Structure ===
"""
Example Node Instances:

User Node:
{
    "id": "u1",
    "type": "User",
    "name": "user_A"
}

Knowledge Node (Object Semantics):
{
    "id": "k1",
    "type": "Knowledge",
    "subtype": "object_semantics",
    "alias": "my favorite cup"
}

Knowledge Node (User Pattern):
{
    "id": "k2",
    "type": "Knowledge",
    "subtype": "user_pattern",
    "alias": "late night snack setup"
}

Pattern Node:
{
    "id": "p1",
    "type": "Pattern",
    "name": "place",
    "args": ["cup", "on", "livingroom_table"]
}

Object Node (Type):
{
    "id": "o1",
    "type": "Object",
    "name": "cup",
    "granularity": "type"
}

Object Node (Instance):
{
    "id": "o2",
    "type": "Object",
    "name": "red dog image cup",
    "granularity": "instance"
}

Location Node:
{
    "id": "l1",
    "type": "Location",
    "name": "livingroom_table",
    "expression": "on the table"
}

Example Edge Relations:

Hierarchical Edges:
- User → Knowledge: "owns"
- Knowledge → Pattern: "entails"
- Pattern → Object: "target"
- Pattern → Location: "target"
- Knowledge → Object: "alias_of", "target_object"
- Knowledge → Location: "target_location"
- Pattern → Location: "source_location"
- Object → Object: "next_to"

Temporal Edges:
- Pattern → Pattern: "before"
"""
