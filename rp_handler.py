import os
import re
import runpod
from transformers import pipeline

# Model ve ayarlar Runpod endpoint'inin Environment Variables kısmından değiştirilebilir
MODEL_ID = os.environ.get("MODEL_ID", "zai-org/GLM-5.3-Flash")
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "true").lower() in ("1", "true", "yes")

# Modeli handler dışında yüklemek önemli (her request'te yeniden yüklemesin)
pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=TRUST_REMOTE_CODE,
)

def _extract_python(code: str) -> str:
    """
    Model bazen ```python ...``` döndürür; varsa fence'leri temizler.
    """
    if code is None:
        return ""
    code = code.strip()

    m = re.search(r"```(?:python)?\s*(.*?)```", code, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return code

def handler(job):
    job_input = job.get("input", {})  # Runpod job formatı
    prompt = job_input.get("prompt", "")
    filename = job_input.get("filename", "solution.py")
    gen = job_input.get("generation", {}) or {}

    # generation parametreleri (defaultlar)
    max_new_tokens = int(gen.get("max_new_tokens", 800))
    temperature = float(gen.get("temperature", 0.2))
    top_p = float(gen.get("top_p", 0.9))

    system_instruction = (
        "You are a code generator. Return ONLY valid Python code. "
        "Do not include markdown, backticks, or explanations inside the code."
    )

    final_prompt = f"{system_instruction}\n\nTASK:\n{prompt}\n\nPYTHON CODE:\n"

    # temperature 0 ise greedy decoding (sampling 0 sıcaklıkla hata verir)
    sample_kwargs = (
        {"do_sample": True, "temperature": temperature, "top_p": top_p}
        if temperature > 0
        else {"do_sample": False}
    )

    out = pipe(
        final_prompt,
        max_new_tokens=max_new_tokens,
        return_full_text=False,
        **sample_kwargs,
    )

    # transformers pipeline çıktısı liste/dict
    text = out[0]["generated_text"] if out and isinstance(out, list) else str(out)
    py = _extract_python(text)

    report_md = (
        "## GLM-5.3-Flash Worker Report\n"
        f"- model: `{MODEL_ID}`\n"
        f"- filename: `{filename}`\n"
        "### Generation params\n"
        f"- max_new_tokens: {max_new_tokens}\n"
        f"- temperature: {temperature}\n"
        f"- top_p: {top_p}\n"
        "### Notes\n"
        "- Returned `py` is the model output with markdown fences stripped (if present).\n"
    )

    return {
        "filename": filename,
        "py": py,
        "report_md": report_md
    }

runpod.serverless.start({"handler": handler})
