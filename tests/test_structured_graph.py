from pydantic import BaseModel

from structured_graph import StructuredOutputGraph


class LessonRoute(BaseModel):
    topic: str
    lesson_type: str


class FakeClient:
    def structured(self, messages, output_model, schema_name, model=None, **kwargs):
        return output_model(topic="hybrid search", lesson_type="Independent Practice")


def test_structured_output_graph_invokes_langgraph_workflow() -> None:
    graph = StructuredOutputGraph(
        client=FakeClient(),
        output_model=LessonRoute,
        schema_name="lesson_route",
        system_prompt="Classify lesson requests.",
    )

    result = graph.invoke("I need students to tune keyword and semantic blending.")

    assert result == LessonRoute(topic="hybrid search", lesson_type="Independent Practice")
