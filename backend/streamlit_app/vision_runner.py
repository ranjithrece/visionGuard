from graph.pipeline import VisionGuardGraph

graph = VisionGuardGraph().build()

def run_vision_guard_pipeline(image_path:str) -> dict:
    input_state = {
    "img_path": image_path,
    "caption": "",
    "query": "",
    "results": [],
    "summary": "",
    "evaluation": ""
}

    output = graph.invoke(input_state)
    return output