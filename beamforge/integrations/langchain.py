"""LangChain-compatible LLM wrapper (works even without LangChain).

Defines the minimal `_generate`/`_llm_type` surface LangChain expects; if the
library is absent it still constructs, so the module never hard-fails.
"""
from ..model.engine import BeamEngine


class BeamForgeLLM:
    def __init__(self, backend=None, config=None):
        self.engine = BeamEngine(backend=backend, config=config)

    @property
    def _llm_type(self):
        return "beamforge"

    def _generate(self, prompts, stop=None, **kw):
        generations = []
        for p in prompts:
            out = self.engine.answer(p, **kw)
            generations.append([{"text": out["answer"], "generation_info": out}])
        return type("LLMResult", (), {"generations": generations})()

    def __call__(self, prompt, **kw):
        return self.engine.answer(prompt, **kw)["answer"]
