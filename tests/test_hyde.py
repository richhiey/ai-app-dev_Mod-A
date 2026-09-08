from hyde import HyDEQuery, HyDERewriter
from models import ChatModel


class FakeStructuredClient:
    def __init__(self):
        self.calls = []

    def structured(self, messages, output_model, schema_name, model=None, **kwargs):
        self.calls.append(
            {"messages": messages, "output_model": output_model, "schema_name": schema_name, "model": model}
        )
        return output_model(
            original_query="Where does naive RAG break?",
            rewritten_query="naive RAG failure modes chunk quality retrieval context overflow",
            hypothetical_document="Naive RAG often fails from weak chunks, noisy retrieval, and context overflow.",
        )


def test_hyde_rewriter_uses_enabled_course_model_and_schema() -> None:
    client = FakeStructuredClient()
    rewrite = HyDERewriter(client).rewrite("Where does naive RAG break?")

    assert isinstance(rewrite, HyDEQuery)
    assert "context overflow" in rewrite.hypothetical_document
    assert client.calls[0]["model"] == ChatModel.GEMINI_25_FLASH_LITE
    assert client.calls[0]["schema_name"] == "hyde_query"
