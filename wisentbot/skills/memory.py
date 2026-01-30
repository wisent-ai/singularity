#!/usr/bin/env python3
"""
Memory Skill - Persistent AI memory using cognee.

Provides agents with:
- Knowledge graph memory that persists across sessions
- Vector + graph hybrid search
- Multiple search modes (graph completion, RAG, summaries, code)
- Memory of past conversations and experiences
- Shared memory between agents

See: https://github.com/topoteretes/cognee
"""

from typing import Dict, List, Optional, Any
from .base import Skill, SkillManifest, SkillAction, SkillResult

# Check if cognee is available
HAS_COGNEE = False
COGNEE_VERSION = None
try:
    import cognee
    COGNEE_VERSION = getattr(cognee, '__version__', 'unknown')
    from cognee.api.v1.search import SearchType
    HAS_COGNEE = True
except ImportError:
    pass

# Search types available
SEARCH_TYPES = {
    "graph": "Natural language Q&A with full graph context (recommended)",
    "rag": "Traditional RAG using document chunks",
    "chunks": "Raw text segments matching query",
    "summaries": "Pre-generated hierarchical summaries",
    "code": "Code-specific search with syntax understanding",
}


class MemorySkill(Skill):
    """
    Persistent memory skill using cognee's knowledge graph.

    Agents can:
    - Store memories (text, conversations, files)
    - Build knowledge graphs from memories
    - Search memories by meaning and relationships
    - Share memories between agent sessions
    """

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)
        self._initialized = False
        self._agent_name: Optional[str] = None
        self._dataset_prefix: str = "wisentbot"

    def set_agent_context(self, agent_name: str, dataset_prefix: str = "wisentbot"):
        """Set the agent context for memory isolation."""
        self._agent_name = agent_name
        self._dataset_prefix = dataset_prefix

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="memory",
            name="Cognee Memory",
            version="1.0.0",
            category="memory",
            description="Persistent AI memory with knowledge graphs",
            actions=[
                # === Core Memory Operations ===
                SkillAction(
                    name="remember",
                    description="Add something to memory (text, conversation, experience)",
                    parameters={
                        "content": {
                            "type": "string",
                            "required": True,
                            "description": "Content to remember"
                        },
                        "dataset": {
                            "type": "string",
                            "required": False,
                            "description": "Dataset name (default: agent's dataset)"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="remember_file",
                    description="Add a file to memory",
                    parameters={
                        "path": {
                            "type": "string",
                            "required": True,
                            "description": "Path to file"
                        },
                        "dataset": {
                            "type": "string",
                            "required": False,
                            "description": "Dataset name"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="cognify",
                    description="Process memories into knowledge graph (run after adding memories)",
                    parameters={
                        "dataset": {
                            "type": "string",
                            "required": False,
                            "description": "Dataset to process (default: all)"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="memify",
                    description="Add memory algorithms to the graph (run after cognify)",
                    parameters={},
                    estimated_cost=0,
                ),
                # === Search ===
                SkillAction(
                    name="recall",
                    description="Search memories using natural language",
                    parameters={
                        "query": {
                            "type": "string",
                            "required": True,
                            "description": "What to search for"
                        },
                        "search_type": {
                            "type": "string",
                            "required": False,
                            "description": "Type: graph, rag, chunks, summaries, code (default: graph)"
                        },
                        "top_k": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of results (default: 10)"
                        },
                        "dataset": {
                            "type": "string",
                            "required": False,
                            "description": "Search specific dataset"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="recall_context",
                    description="Get relevant context for a conversation (returns raw context, no LLM)",
                    parameters={
                        "query": {
                            "type": "string",
                            "required": True,
                            "description": "Current conversation context"
                        },
                        "top_k": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of results (default: 5)"
                        }
                    },
                    estimated_cost=0,
                ),
                # === Management ===
                SkillAction(
                    name="forget",
                    description="Delete memories from a dataset",
                    parameters={
                        "dataset": {
                            "type": "string",
                            "required": True,
                            "description": "Dataset to delete"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="prune",
                    description="Clean up and optimize memory storage",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="datasets",
                    description="List all memory datasets",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="status",
                    description="Get memory system status",
                    parameters={},
                    estimated_cost=0,
                ),
                # === Conversation Memory ===
                SkillAction(
                    name="remember_conversation",
                    description="Store a conversation exchange for future reference",
                    parameters={
                        "user_message": {
                            "type": "string",
                            "required": True,
                            "description": "What the user said"
                        },
                        "agent_response": {
                            "type": "string",
                            "required": True,
                            "description": "What you responded"
                        },
                        "outcome": {
                            "type": "string",
                            "required": False,
                            "description": "Outcome: success, failure, or neutral"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="remember_learning",
                    description="Store something you learned for future reference",
                    parameters={
                        "topic": {
                            "type": "string",
                            "required": True,
                            "description": "What topic this relates to"
                        },
                        "learning": {
                            "type": "string",
                            "required": True,
                            "description": "What you learned"
                        },
                        "source": {
                            "type": "string",
                            "required": False,
                            "description": "Where you learned this"
                        }
                    },
                    estimated_cost=0,
                ),
                # === Visualization ===
                SkillAction(
                    name="visualize",
                    description="Generate a visualization of the knowledge graph",
                    parameters={
                        "output_path": {
                            "type": "string",
                            "required": False,
                            "description": "Path to save visualization"
                        }
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        """Check if cognee is available."""
        return HAS_COGNEE

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if not HAS_COGNEE:
            return SkillResult(
                success=False,
                message="cognee not installed. Run: pip install cognee"
            )

        handlers = {
            "remember": self._remember,
            "remember_file": self._remember_file,
            "cognify": self._cognify,
            "memify": self._memify,
            "recall": self._recall,
            "recall_context": self._recall_context,
            "forget": self._forget,
            "prune": self._prune,
            "datasets": self._datasets,
            "status": self._status,
            "remember_conversation": self._remember_conversation,
            "remember_learning": self._remember_learning,
            "visualize": self._visualize,
        }

        handler = handlers.get(action)
        if handler:
            return await handler(params)
        return SkillResult(success=False, message=f"Unknown action: {action}")

    def _get_dataset_name(self, dataset: Optional[str] = None) -> str:
        """Get the full dataset name with prefix."""
        if dataset:
            return f"{self._dataset_prefix}_{dataset}"
        if self._agent_name:
            return f"{self._dataset_prefix}_{self._agent_name}"
        return self._dataset_prefix

    # === Handlers ===

    async def _remember(self, params: Dict) -> SkillResult:
        """Add content to memory."""
        content = params.get("content", "").strip()
        dataset = params.get("dataset")

        if not content:
            return SkillResult(success=False, message="Content required")

        try:
            dataset_name = self._get_dataset_name(dataset)
            await cognee.add(content, dataset_name)

            return SkillResult(
                success=True,
                message=f"Remembered ({len(content)} chars) in '{dataset_name}'",
                data={
                    "dataset": dataset_name,
                    "content_length": len(content),
                    "hint": "Run memory:cognify to process into knowledge graph",
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Remember failed: {e}")

    async def _remember_file(self, params: Dict) -> SkillResult:
        """Add a file to memory."""
        path = params.get("path", "").strip()
        dataset = params.get("dataset")

        if not path:
            return SkillResult(success=False, message="Path required")

        try:
            dataset_name = self._get_dataset_name(dataset)
            await cognee.add(path, dataset_name)

            return SkillResult(
                success=True,
                message=f"Remembered file '{path}' in '{dataset_name}'",
                data={
                    "dataset": dataset_name,
                    "path": path,
                    "hint": "Run memory:cognify to process into knowledge graph",
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Remember file failed: {e}")

    async def _cognify(self, params: Dict) -> SkillResult:
        """Process memories into knowledge graph."""
        dataset = params.get("dataset")

        try:
            if dataset:
                dataset_name = self._get_dataset_name(dataset)
                await cognee.cognify(datasets=[dataset_name])
            else:
                await cognee.cognify()

            return SkillResult(
                success=True,
                message="Knowledge graph built from memories",
                data={
                    "dataset": dataset if dataset else "all",
                    "hint": "Run memory:memify to add memory algorithms",
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Cognify failed: {e}")

    async def _memify(self, params: Dict) -> SkillResult:
        """Add memory algorithms to the graph."""
        try:
            await cognee.memify()

            return SkillResult(
                success=True,
                message="Memory algorithms added to graph",
                data={
                    "hint": "Memories are now searchable with memory:recall",
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Memify failed: {e}")

    async def _recall(self, params: Dict) -> SkillResult:
        """Search memories."""
        query = params.get("query", "").strip()
        search_type = params.get("search_type", "graph").lower()
        top_k = params.get("top_k", 10)
        dataset = params.get("dataset")

        if not query:
            return SkillResult(success=False, message="Query required")

        if search_type not in SEARCH_TYPES:
            return SkillResult(
                success=False,
                message=f"Unknown search type: {search_type}. Available: {list(SEARCH_TYPES.keys())}"
            )

        try:
            # Map search type to cognee SearchType
            type_map = {
                "graph": SearchType.GRAPH_COMPLETION,
                "rag": SearchType.RAG_COMPLETION,
                "chunks": SearchType.CHUNKS,
                "summaries": SearchType.SUMMARIES,
                "code": SearchType.CODE,
            }
            cognee_type = type_map.get(search_type, SearchType.GRAPH_COMPLETION)

            # Build search kwargs
            kwargs = {
                "query_text": query,
                "query_type": cognee_type,
                "top_k": top_k,
            }

            if dataset:
                dataset_name = self._get_dataset_name(dataset)
                kwargs["datasets"] = [dataset_name]

            results = await cognee.search(**kwargs)

            # Format results
            if isinstance(results, list):
                formatted = [
                    {
                        "content": str(r.get("content", r) if isinstance(r, dict) else r),
                        "score": r.get("score", None) if isinstance(r, dict) else None,
                    }
                    for r in results[:top_k]
                ]
            else:
                formatted = [{"content": str(results)}]

            return SkillResult(
                success=True,
                message=f"Found {len(formatted)} memories for '{query[:50]}...'",
                data={
                    "query": query,
                    "search_type": search_type,
                    "results": formatted,
                    "count": len(formatted),
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Recall failed: {e}")

    async def _recall_context(self, params: Dict) -> SkillResult:
        """Get raw context without LLM processing."""
        query = params.get("query", "").strip()
        top_k = params.get("top_k", 5)

        if not query:
            return SkillResult(success=False, message="Query required")

        try:
            results = await cognee.search(
                query_text=query,
                query_type=SearchType.CHUNKS,
                top_k=top_k,
                only_context=True,
            )

            # Extract just the text content
            context_pieces = []
            if isinstance(results, list):
                for r in results:
                    if isinstance(r, dict):
                        context_pieces.append(r.get("content", str(r)))
                    else:
                        context_pieces.append(str(r))

            return SkillResult(
                success=True,
                message=f"Retrieved {len(context_pieces)} context pieces",
                data={
                    "query": query,
                    "context": context_pieces,
                    "count": len(context_pieces),
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Recall context failed: {e}")

    async def _forget(self, params: Dict) -> SkillResult:
        """Delete a dataset."""
        dataset = params.get("dataset", "").strip()

        if not dataset:
            return SkillResult(success=False, message="Dataset name required")

        try:
            dataset_name = self._get_dataset_name(dataset)
            await cognee.delete(dataset_name)

            return SkillResult(
                success=True,
                message=f"Deleted dataset '{dataset_name}'",
                data={"dataset": dataset_name}
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Forget failed: {e}")

    async def _prune(self, params: Dict) -> SkillResult:
        """Clean up memory storage."""
        try:
            await cognee.prune()

            return SkillResult(
                success=True,
                message="Memory storage cleaned up"
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Prune failed: {e}")

    async def _datasets(self, params: Dict) -> SkillResult:
        """List all datasets."""
        try:
            datasets_info = await cognee.datasets()

            # Filter to our prefix
            our_datasets = [
                d for d in datasets_info
                if isinstance(d, dict) and d.get("name", "").startswith(self._dataset_prefix)
            ] if isinstance(datasets_info, list) else []

            return SkillResult(
                success=True,
                message=f"Found {len(our_datasets)} datasets",
                data={
                    "datasets": our_datasets,
                    "prefix": self._dataset_prefix,
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"List datasets failed: {e}")

    async def _status(self, params: Dict) -> SkillResult:
        """Get memory system status."""
        return SkillResult(
            success=True,
            message="Memory system status",
            data={
                "cognee_installed": HAS_COGNEE,
                "cognee_version": COGNEE_VERSION,
                "agent_name": self._agent_name,
                "dataset_prefix": self._dataset_prefix,
                "search_types": SEARCH_TYPES,
            }
        )

    async def _remember_conversation(self, params: Dict) -> SkillResult:
        """Store a conversation exchange."""
        user_message = params.get("user_message", "").strip()
        agent_response = params.get("agent_response", "").strip()
        outcome = params.get("outcome", "neutral")

        if not user_message or not agent_response:
            return SkillResult(success=False, message="user_message and agent_response required")

        try:
            # Format as structured conversation memory
            content = f"""CONVERSATION MEMORY
User: {user_message}
Agent: {agent_response}
Outcome: {outcome}
"""
            dataset_name = self._get_dataset_name("conversations")
            await cognee.add(content, dataset_name)

            return SkillResult(
                success=True,
                message=f"Remembered conversation (outcome: {outcome})",
                data={
                    "dataset": dataset_name,
                    "outcome": outcome,
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Remember conversation failed: {e}")

    async def _remember_learning(self, params: Dict) -> SkillResult:
        """Store a learning."""
        topic = params.get("topic", "").strip()
        learning = params.get("learning", "").strip()
        source = params.get("source", "experience")

        if not topic or not learning:
            return SkillResult(success=False, message="topic and learning required")

        try:
            # Format as structured learning memory
            content = f"""LEARNING MEMORY
Topic: {topic}
Learning: {learning}
Source: {source}
"""
            dataset_name = self._get_dataset_name("learnings")
            await cognee.add(content, dataset_name)

            return SkillResult(
                success=True,
                message=f"Remembered learning about '{topic}'",
                data={
                    "dataset": dataset_name,
                    "topic": topic,
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Remember learning failed: {e}")

    async def _visualize(self, params: Dict) -> SkillResult:
        """Visualize the knowledge graph."""
        output_path = params.get("output_path")

        try:
            if output_path:
                await cognee.visualize_graph(output_path)
                return SkillResult(
                    success=True,
                    message=f"Graph visualization saved to {output_path}",
                    data={"path": output_path}
                )
            else:
                # Start visualization server
                await cognee.start_visualization_server()
                return SkillResult(
                    success=True,
                    message="Visualization server started",
                    data={"hint": "Open browser to view knowledge graph"}
                )
        except Exception as e:
            return SkillResult(success=False, message=f"Visualize failed: {e}")
