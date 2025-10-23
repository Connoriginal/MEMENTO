# Semantic Memory Graph Encoder

User의 instruction과 trajectory를 입력받아서 structured semantic graph로 encoding하는 시스템입니다.

## 주요 구성 요소

### 1. Graph Schema (`graph_schema.py`)

그래프의 노드와 엣지 타입을 정의합니다.

**노드 타입:**
- `User`: 사용자
- `Knowledge`: 시맨틱 지식 (object_semantics, user_pattern)
- `Action`: 구체적인 행동
- `Object`: 객체 (type/instance 레벨)
- `Location`: 위치

**엣지 타입:**
- `owns`: User → Knowledge
- `entails`: Knowledge → Action
- `target`: Action → Object/Location
- `alias_of`: Knowledge → Object
- `before`: Action → Action (시간 순서)

### 2. Graph Encoder (`graph_encoder.py`)

#### GraphEncoder
LLM을 활용하여 자연어 instruction과 trajectory를 semantic graph로 변환합니다.

```python
from graph_encoder import GraphEncoder, EncodingInput, TrajectoryStep

# 1. Encoder 초기화
encoder = GraphEncoder(api_key="your-openai-api-key")

# 2. 입력 데이터 준비
trajectory = [
    TrajectoryStep(action="navigate_to_object", objects=["book"], locations=["living_room"]),
    TrajectoryStep(action="pick_up", objects=["book"], locations=["living_room_table"]),
    TrajectoryStep(action="navigate_to_location", objects=["book"], locations=["bedroom"]),
    TrajectoryStep(action="place_object", objects=["book"], locations=["bedroom_dresser"])
]

input_data = EncodingInput(
    user_id="user_A",
    instruction="Move the white book with bookmark from living room table to bedroom dresser",
    trajectory=trajectory,
    episode_id="ep_001",
    scene_id="scene_001"
)

# 3. Graph 생성
graph = encoder.encode_to_graph(input_data)
```

#### GraphComparator
기존 그래프와 새 그래프를 비교하여 add/update/merge를 결정합니다.

```python
from graph_encoder import GraphComparator

comparator = GraphComparator(similarity_threshold=0.8)
action, details = comparator.compare_and_decide(existing_graph, new_graph)

print(f"Decision: {action}")  # "add", "update", "merge"
```

#### GraphManager
그래프 파일을 관리하고 embedding을 위한 데이터를 준비합니다.

```python
from graph_encoder import GraphManager

# 1. Manager 초기화 (graph.json 파일 경로)
manager = GraphManager("output/semantic_graph.json")

# 2. 새로운 입력 처리
result = manager.process_new_input(input_data, encoder)

print(f"Action taken: {result['action']}")
print(f"Final nodes: {result.get('final_nodes', 'N/A')}")

# 3. Embedding 데이터 준비
embedding_data = manager.prepare_for_embedding("output/embedding_data.json")
```

## 사용 시나리오

### 1. 첫 번째 경험 추가
```python
# 새로운 그래프 생성
result = manager.process_new_input(input_data, encoder)
# result['action'] == "create"
```

### 2. 유사하지 않은 새로운 경험 추가
```python
# 기존 그래프에 새로운 노드들 추가
result = manager.process_new_input(different_input, encoder)
# result['action'] == "add"
```

### 3. 유사한 경험으로 업데이트
```python
# 기존 유사한 노드들과 연결/업데이트
result = manager.process_new_input(similar_input, encoder)
# result['action'] == "update"
```

### 4. 복잡한 병합
```python
# 많은 유사한 노드들이 있는 경우 지능적 병합
result = manager.process_new_input(complex_input, encoder)
# result['action'] == "merge"
```

## 출력 형태

### 1. Semantic Graph JSON
```json
{
  "nodes": [
    {
      "id": "u_user1",
      "type": "User",
      "name": "user_A"
    },
    {
      "id": "k_book_semantics",
      "type": "Knowledge",
      "subtype": "object_semantics",
      "alias": "white book with bookmark"
    },
    {
      "id": "a_move_book",
      "type": "Action",
      "name": "move",
      "args": ["book", "from", "living_room", "to", "bedroom"]
    }
  ],
  "edges": [
    {
      "source_id": "u_user1",
      "target_id": "k_book_semantics",
      "relation": "owns"
    },
    {
      "source_id": "k_book_semantics",
      "target_id": "a_move_book",
      "relation": "entails"
    }
  ]
}
```

### 2. Embedding 준비 데이터
```json
{
  "node_texts": {
    "u_user1": "User: user_A",
    "k_book_semantics": "Knowledge: white book with bookmark",
    "a_move_book": "Action: move with args book, from, living_room, to, bedroom"
  },
  "edge_descriptions": [
    "User: user_A owns Knowledge: white book with bookmark",
    "Knowledge: white book with bookmark entails Action: move with args book, from, living_room, to, bedroom"
  ],
  "graph_summary": {
    "total_nodes": 10,
    "total_edges": 15,
    "node_types": {"User": 1, "Knowledge": 2, "Action": 4, "Object": 2, "Location": 1},
    "edge_types": {"owns": 2, "entails": 2, "target": 8, "alias_of": 2, "before": 1}
  }
}
```

## 테스트 실행

```bash
# 전체 파이프라인 테스트 (모킹된 LLM 사용)
python test_graph_encoder.py

# 실제 OpenAI API 사용 테스트
# 1. OpenAI API key 설정
export OPENAI_API_KEY="your-api-key"

# 2. 실제 API로 테스트
python test_with_real_llm.py  # (별도 구현 필요)
```

## 다음 단계 개선사항

1. **실제 LLM API 연동**: OpenAI API key로 실제 테스트
2. **Embedding 기반 유사도**: 텍스트 유사도를 embedding 기반으로 개선
3. **Graph Merge 로직**: 더 정교한 노드 병합 알고리즘
4. **실제 데이터 테스트**: HabitatLLM trajectory 데이터로 검증
5. **성능 최적화**: 대용량 그래프 처리 최적화
6. **시각화**: NetworkX, Cytoscape 등을 통한 그래프 시각화

## 장점

✅ **모듈화된 구조**: 각 컴포넌트가 독립적으로 작동
✅ **확장 가능**: 새로운 노드/엣지 타입 쉽게 추가 가능
✅ **LLM 기반**: 자연어를 structured data로 자동 변환
✅ **증분 업데이트**: 기존 그래프에 새로운 정보 점진적 추가
✅ **Embedding 준비**: 검색/RAG을 위한 데이터 자동 생성
✅ **JSON 기반**: 저장/로드가 간단하고 다른 시스템과 호환성 좋음

이제 사용자의 instruction과 trajectory를 semantic graph로 encoding하고, 이를 지속적으로 관리하며 embedding을 위한 준비까지 완료하는 전체 파이프라인이 구축되었습니다!

