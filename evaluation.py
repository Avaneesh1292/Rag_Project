from observability import observe  # version-safe, no-op fallback included


@observe(name="rag_evaluation")
def run_evaluation(eval_data, retriever, rag_fn):
    results = []

    for item in eval_data:
        output = rag_fn(item["question"], retriever)
        docs = output["docs"]
        expected = item["expected"].lower()

        recall = int(any(expected in d.page_content.lower() for d in docs[:5]))
        mrr = 0
        for i, d in enumerate(docs):
            if expected in d.page_content.lower():
                mrr = 1 / (i + 1)
                break

        results.append({
            "recall@5": recall,
            "mrr": mrr
        })

    return results
